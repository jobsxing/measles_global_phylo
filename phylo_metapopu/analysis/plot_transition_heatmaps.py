"""Heatmap extracted from saptiotemporal_new.ipynb cells 12 and 14 (one-based)."""
from pathlib import Path
from metapopulation_common import load_data

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
_, _, _, _, df_jump_C1 = load_data()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import matplotlib as mpl

REGIONS = ["Americas", "Europe", "Asia", "Oceania", "Endemic"]

REGION_LABELS = {
    "Americas": "Eliminated\nAmericas",
    "Europe":   "Eliminated\nEurope",
    "Asia":     "Eliminated\nAsia",
    "Oceania":  "Eliminated\nOceania",
    "Endemic":  "Endemic\nregion"
}

CUSTOM_COLORS = [
    "#780000",
    "#c1121f",
    "#fdf0d5",
    "#003049",
    "#669bbc"
]
cmap = LinearSegmentedColormap.from_list(
    "custom_continuous",
    CUSTOM_COLORS[::-1],
    N=256
)

def parse_location(loc: str):
    s = str(loc)
    parts = s.split("_")
    if len(parts) == 2 and parts[1].lower() == "endemic":
        return int(parts[0]), "Endemic"
    if len(parts) >= 3 and parts[-1].lower() == "eliminated":
        return int(parts[0]), parts[1]
    return None, None

TIME_WINDOWS_DEC = {
    "Pre-resurgence": (2011.000, 2018.083),
    "Resurgence":     (2018.083, 2020.250),
    "Pandemic":       (2020.250, 2023.333),
    "Post-pandemic":  (2023.333, 2025.667),
}

df_jump_plot = df_jump_C1.copy()

df_jump_plot["time_num"] = pd.to_numeric(df_jump_plot["time"], errors="coerce")
df_jump_plot = df_jump_plot[df_jump_plot["time_num"].notna()].copy()

df_jump_plot[["startYear", "startRegion"]] = df_jump_plot["startLocation"].apply(
    lambda s: pd.Series(parse_location(s))
)
df_jump_plot[["endYear", "endRegion"]] = df_jump_plot["endLocation"].apply(
    lambda s: pd.Series(parse_location(s))
)

df_jump_plot = df_jump_plot[
    df_jump_plot["startRegion"].isin(REGIONS) &
    df_jump_plot["endRegion"].isin(REGIONS) &
    (df_jump_plot["startRegion"] != df_jump_plot["endRegion"])
].copy()


def make_heatmap_matrix_from_raw_mean_across_trees(df, tmin, tmax):
    """
    返回：
      mat:      行列热图矩阵，数值为 annual mean transitions
      flow_avg: 长表，包含跨树平均后的 total_mean_jumps 和 annual_mean

    说明：
      1) 保留所有 startRegion -> endRegion 组合；
      2) 包括所有自循环：
         Americas -> Americas
         Europe   -> Europe
         Asia     -> Asia
         Oceania  -> Oceania
         Endemic  -> Endemic
      3) 0 或 NA 在绘图时仍为空白；
      4) annual_mean = total_mean_jumps / time-window length。
    """

    sub = df[
        (df["time_num"] >= tmin) &
        (df["time_num"] < tmax)
    ].copy()

    years_span = max(tmax - tmin, 1e-9)

    if sub.empty:
        mat = pd.DataFrame(
            np.nan,
            index=REGIONS,
            columns=REGIONS
        )

        flow_avg = pd.DataFrame(columns=[
            "startRegion",
            "endRegion",
            "total_mean_jumps",
            "annual_mean"
        ])

        return mat, flow_avg

    per_tree = (
        sub
        .groupby(["treeId", "startRegion", "endRegion"], as_index=False)
        .size()
        .rename(columns={"size": "n_jumps_tree"})
    )

    tree_ids = pd.DataFrame({
        "treeId": sorted(sub["treeId"].unique())
    })

    all_pairs = pd.MultiIndex.from_product(
        [REGIONS, REGIONS],
        names=["startRegion", "endRegion"]
    ).to_frame(index=False)


    full = (
        tree_ids.assign(key=1)
        .merge(all_pairs.assign(key=1), on="key")
        .drop(columns="key")
    )

    per_tree_full = full.merge(
        per_tree,
        on=["treeId", "startRegion", "endRegion"],
        how="left"
    )

    per_tree_full["n_jumps_tree"] = (
        per_tree_full["n_jumps_tree"]
        .fillna(0)
    )

    flow_avg = (
        per_tree_full
        .groupby(["startRegion", "endRegion"], as_index=False)
        .agg(
            total_mean_jumps=("n_jumps_tree", "mean")
        )
    )

    flow_avg["annual_mean"] = (
        flow_avg["total_mean_jumps"] / years_span
    )

    mat = flow_avg.pivot(
        index="startRegion",
        columns="endRegion",
        values="annual_mean"
    )

    mat = mat.reindex(
        index=REGIONS,
        columns=REGIONS
    )


    return mat, flow_avg


HEATMAPS = {}
FLOW_AVG_BY_PERIOD = {}

for label, (tmin, tmax) in TIME_WINDOWS_DEC.items():
    mat, flow_avg = make_heatmap_matrix_from_raw_mean_across_trees(
        df_jump_plot,
        tmin,
        tmax
    )

    HEATMAPS[label] = mat
    FLOW_AVG_BY_PERIOD[label] = flow_avg


valid_max = []

for m in HEATMAPS.values():
    vals = m.values.astype(float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    if len(vals) > 0:
        valid_max.append(np.nanmax(vals))

vmax = max(valid_max) if valid_max else 1.0
vmin = 0.0

norm = Normalize(
    vmin=vmin,
    vmax=vmax
)


def draw_manual_heatmap(ax, mat, title, show_y=False):
    nrow, ncol = mat.shape

    ax.set_facecolor("white")

    for i in range(nrow):
        for j in range(ncol):
            val = mat.iloc[i, j]

            facecolor = "white"

            if pd.notna(val) and val > 0:
                facecolor = cmap(norm(val))

            rect = Rectangle(
                (j, i),
                1,
                1,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.5
            )

            ax.add_patch(rect)

            if pd.notna(val) and val > 0:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white"
                )

    ax.set_xlim(0, ncol)
    ax.set_ylim(nrow, 0)
    ax.set_aspect("equal")

    ax.set_title(
        title,
        fontsize=11,
        pad=8
    )

    ax.set_xticks(np.arange(ncol) + 0.5)
    ax.set_xticklabels(
        [REGION_LABELS[r] for r in REGIONS],
        rotation=0
    )
    ax.set_xlabel("To region")

    ax.set_yticks(np.arange(nrow) + 0.5)

    if show_y:
        ax.set_yticklabels(
            [REGION_LABELS[r] for r in REGIONS],
            rotation=0
        )

        ax.set_ylabel("From region")

        ax.text(
            -1.15,
            nrow / 2,
            "Source",
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transData
        )

    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)


fig, axes = plt.subplots(
    nrows=1,
    ncols=4,
    figsize=(17, 4.4),
    sharey=True
)

fig.subplots_adjust(
    left=0.10,
    right=0.99,
    bottom=0.22,
    top=0.88,
    wspace=0.04
)

for k, (ax, (label, mat)) in enumerate(zip(axes, HEATMAPS.items())):
    draw_manual_heatmap(
        ax=ax,
        mat=mat,
        title=label,
        show_y=(k == 0)
    )


sm = mpl.cm.ScalarMappable(
    cmap=cmap,
    norm=norm
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    shrink=0.85,
    pad=0.02
)

cbar.set_label(
    "Annual mean transitions",
    fontsize=11
)


out_pdf = OUT_DIR / "transition_heatmaps_epochs_exact_treeavg_with_self_loops.pdf"

fig.savefig(
    out_pdf,
    format="pdf",
    bbox_inches="tight",
    facecolor="white"
)

print("Saved to:", out_pdf)

plt.show()
