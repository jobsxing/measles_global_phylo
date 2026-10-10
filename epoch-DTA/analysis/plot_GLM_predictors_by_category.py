"""
Plot the four requested revision GLM predictor figures, grouped by theme.
Inputs are the three per-predictor summary CSVs stored beside this script.

Run with:
  python plot_GLM_predictors_by_category.py
"""

import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei", "SimSun"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
PRIOR_Q = 0.05
PRIOR_ODDS = PRIOR_Q / (1 - PRIOR_Q)
BF_REFERENCE = (3, 10, 100)
EPOCHS = [1, 2, 3, 4]
EPOCH_NAMES = {
    1: "Epoch 1: Pre-resurgence",
    2: "Epoch 2: Measles resurgence",
    3: "Epoch 3: COVID-19 pandemic",
    4: "Epoch 4: Post-pandemic",
}

PREDICTOR_GROUPS = [
    ("Mobility &\nconnectivity", [1, 4, 5, 6]),
    ("Vaccination", [2, 3]),
    ("Demography\n& economy", [7, 8, 9, 10]),
    ("Forced\ndisplacement", [13, 14, 15]),
    ("Sampling\neffort", [11, 12]),
]
PREDICTOR_GROUP_OF = {
    k: name.replace("\n", " ") for name, members in PREDICTOR_GROUPS for k in members
}

BAND_COLOR = "#F0F0F0"
BAR_COLOR = "#D2D2D2"


def group_layout(selected):
    """Return (ordered predictor ids, [(group name, [row positions])])."""
    ordered, groups = [], []
    for name, members in PREDICTOR_GROUPS:
        keep = [k for k in members if k in selected]
        if not keep:
            continue
        groups.append((name, list(range(len(ordered), len(ordered) + len(keep)))))
        ordered.extend(keep)
    leftover = [k for k in sorted(selected) if k not in ordered]
    if leftover:
        groups.append(("Other", list(range(len(ordered), len(ordered) + len(leftover)))))
        ordered.extend(leftover)
    return ordered, groups


def bf_cut(bf_ref):
    odds = PRIOR_ODDS * bf_ref
    return odds / (1 + odds)


def draw_group_bands(ax, groups):
    for gi, (_, positions) in enumerate(groups):
        top = min(positions) - 0.5
        bottom = max(positions) + 0.5
        if gi % 2 == 1:
            ax.axhspan(top, bottom, color=BAND_COLOR, zorder=0)
        if gi:
            ax.axhline(top, color="black", lw=0.5, alpha=0.45, zorder=1.5)


def draw_group_brackets(ax, groups, fontsize=13, x_line=1.03, x_text=1.06):
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for name, positions in groups:
        ax.plot(
            [x_line, x_line], [min(positions) - 0.4, max(positions) + 0.4],
            transform=trans, color="black", lw=1.3, clip_on=False, zorder=6,
        )
        ax.text(
            x_text, (min(positions) + max(positions)) / 2, name,
            transform=trans, ha="left", va="center",
            fontsize=fontsize, fontweight="bold", clip_on=False, zorder=6,
        )


def plot_box_interval(stats, title, out_stem, pip_threshold=None):
    """Bar (E[delta]) + box-interval (beta | delta=1) panels, one pair per epoch."""
    if pip_threshold is None:
        selected = set(stats["predictor"].unique())
    else:
        keep = stats.groupby("predictor")["posterior_inclusion_probability"].max()
        selected = set(keep[keep > pip_threshold].index)
    if not selected:
        print(f"No predictors selected for {out_stem}; skipped.")
        return

    ordered, groups = group_layout(selected)
    n_pred = len(ordered)
    row_of = {k: i for i, k in enumerate(ordered)}
    labels = (
        stats.drop_duplicates("predictor").set_index("predictor")["predictor_label"].to_dict()
    )

    medians = stats.loc[stats["predictor"].isin(selected), "median"].dropna()
    vmin = float(np.floor(medians.min() * 10) / 10)
    vmax = float(np.ceil(medians.max() * 10) / 10)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap("cividis")

    fig = plt.figure(figsize=(21, max(7.5, 0.62 * n_pred + 3.2)), facecolor="white")
    outer = gridspec.GridSpec(1, 4, wspace=0.16, left=0.13, right=0.88, top=0.86, bottom=0.13)

    last_int_ax = None
    for col, e in enumerate(EPOCHS):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[col], width_ratios=[1.0, 2.2], wspace=0.14
        )
        ax_bar = fig.add_subplot(inner[0])
        ax_int = fig.add_subplot(inner[1])
        last_int_ax = ax_int

        sub = stats[stats["epoch"] == e].set_index("predictor")

        for ax in (ax_bar, ax_int):
            draw_group_bands(ax, groups)

        for bf_ref in BF_REFERENCE:
            p_cut = bf_cut(bf_ref)
            ax_bar.axvline(p_cut, ls="-", lw=0.6, color="#BBBBBB", zorder=1)
            ax_bar.text(
                p_cut, -0.035, f"BF={bf_ref}",
                rotation=90, ha="center", va="top", fontsize=11,
                transform=ax_bar.get_xaxis_transform(), clip_on=False,
            )

        for k in ordered:
            if k not in sub.index:
                continue
            row = sub.loc[k]
            i = row_of[k]
            ax_bar.barh(i, row["posterior_inclusion_probability"], height=0.72,
                        color=BAR_COLOR, edgecolor="none", zorder=2)

            if not np.isfinite(row.get("median", np.nan)):
                continue
            ax_int.hlines(i, row["lower95"], row["upper95"], color="black", lw=0.9, zorder=3)
            width = row["upper50"] - row["lower50"]
            ax_int.add_patch(
                mpl.patches.Rectangle(
                    (row["lower50"], i - 0.28), width, 0.56,
                    facecolor=cmap(norm(row["median"])), edgecolor="black",
                    linewidth=0.7, zorder=4,
                )
            )
            ax_int.vlines(row["median"], i - 0.28, i + 0.28, color="black", lw=1.0, zorder=5)

        lo = float(stats.loc[stats["epoch"] == e, "lower95"].min())
        hi = float(stats.loc[stats["epoch"] == e, "upper95"].max())
        pad = 0.06 * (hi - lo) if hi > lo else 0.5
        ax_int.set_xlim(lo - pad, hi + pad)
        ax_int.axvline(0, ls="-", lw=0.7, color="#999999", zorder=1)

        ax_bar.set_xlim(0, 1)
        ax_bar.set_xticks([0, 0.5, 1.0])
        for ax in (ax_bar, ax_int):
            ax.set_ylim(n_pred - 0.5, -0.5)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=12, width=0.6, length=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

        ax_bar.set_title(r"$E[\delta]$", fontsize=15, pad=6)
        ax_int.set_title(r"$\beta \mid \delta=1$", fontsize=15, pad=6)
        mid = ax_bar.get_position().x0 + (ax_int.get_position().x1 - ax_bar.get_position().x0) / 2
        fig.text(mid, 0.90, EPOCH_NAMES[e], ha="center", va="bottom", fontsize=16)

        if col == 0:
            ax_bar.set_yticks(range(n_pred))
            ax_bar.set_yticklabels(
                [textwrap.fill(str(labels.get(k, k)), 18) for k in ordered], fontsize=12.5
            )
            ax_bar.tick_params(axis="y", length=0)

    draw_group_brackets(last_int_ax, groups)

    cax = fig.add_axes([0.795, 0.935, 0.10, 0.014])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([vmin, vmax])
    cb.ax.tick_params(labelsize=10, length=2)
    cb.outline.set_linewidth(0.5)
    cax.set_title("Effect size", fontsize=11, pad=4)

    fig.text(0.13, 0.945, title, ha="left", va="bottom", fontsize=16)

    pdf = BASE_DIR / f"{out_stem}.pdf"
    png = BASE_DIR / f"{out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf.name} / {png.name}  ({n_pred} predictors)")


def main():
    jobs = [
        ("VI_predictor_BF_stats_total_epoch1BF_sorted.csv",
         "VI - all 15 predictors (prior q = 0.05)",
         "VI_epoch1_4_bar_interval_total_by_category", None),
        ("VI_predictor_BF_stats_total_epoch1BF_sorted.csv",
         "VI - predictors with posterior inclusion > 0.5 (prior q = 0.05)",
         "VI_epoch1_4_bar_interval_PP_gt05_by_category", 0.5),
        ("withoutMCV1_predictor_BF_stats_total_epoch1BF_sorted.csv",
         "Sensitivity: MCV2 only - all 15 predictors (prior q = 0.05)",
         "withoutMCV1_epoch1_4_bar_interval_total_by_category", None),
        ("withoutMCV2_predictor_BF_stats_total_epoch1BF_sorted.csv",
         "Sensitivity: MCV1 only - all 15 predictors (prior q = 0.05)",
         "withoutMCV2_epoch1_4_bar_interval_total_by_category", None),
    ]
    for csv_name, title, stem, threshold in jobs:
        stats = pd.read_csv(BASE_DIR / csv_name)
        stats["predictor_group"] = stats["predictor"].map(PREDICTOR_GROUP_OF).fillna("Other")
        plot_box_interval(stats, title, stem, pip_threshold=threshold)


if __name__ == "__main__":
    main()
