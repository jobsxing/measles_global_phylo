"""
Updated two-state Euro resurgence DTA visualization.

This script follows the original Euro_1204/genome notebooks:
  - beast_set*/code*.ipynb for HIPSTR phylogeny plotting with baltic
  - beast_set*/MJ_*.ipynb for Markov-jump invasion/temporal summaries

Run with:
  python visualize_euro_2dta_phylogeny_invasion_mj.py
"""

import os
import re
from datetime import datetime

import baltic as bt
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as clr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2dta")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei", "SimSun"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

STUDIES = [
    {
        "panel": "A",
        "id": "dta2_set1",
        "title": "Two states: 1:1",
        "tree": "dta2_set1.HIPSTR.tree",
        "mj": "mj_history_dta2_set1.csv",
        "label_color": "#108a27",
    },
    {
        "panel": "B",
        "id": "dta2_set2",
        "title": "Two states: 1:1.7",
        "tree": "dta2_set2.HIPSTR.tree",
        "mj": "mj_history_dta2_set2.csv",
        "label_color": "#cf3c1b",
    },
    {
        "panel": "C",
        "id": "dta2_set3",
        "title": "Two states: 1:2.5",
        "tree": "dta2_set3.HIPSTR.tree",
        "mj": "mj_history_dta2_set3.csv",
        "label_color": "#f080a8",
    },
]

TRAIT_NAME = "geography.rate"
STATE_ORDER = ["EUR", "Glo"]
STATE_COLORS = {
    "EUR": "#2d6a4f",
    "Glo": "#8f99ad",
    "ancestor": "#B9B9B9",
}
STATE_LABELS = {
    "EUR": "Europe",
    "Glo": "Global",
}
DIRECTION_ORDER = ["Europe import", "Global import"]
DIRECTION_COLORS = {
    "Europe import": "#2d6a4f",
    "Global import": "#8f99ad",
}


def canon_state(value):
    if value is None:
        return None
    text = re.sub(r"[\s_/\-]+", "", str(value))
    if text.lower() == "glo":
        return "Glo"
    if text.lower() == "eur":
        return "EUR"
    return text


def decimal_date(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25


def extract_tip_dates_from_names(tree_path):
    text = open(tree_path, "r", encoding="utf-8", errors="ignore").read(2_500_000)
    found = re.findall(r"_(\d{4}-\d{2}(?:-\d{2})?)(?:_|:|\[)", text)
    dates = []
    for raw in found:
        if len(raw) == 7:
            raw = raw + "-15"
        try:
            dates.append(datetime.strptime(raw, "%Y-%m-%d"))
        except ValueError:
            pass
    return dates


def latest_tip_anchor(tree_path):
    dates = extract_tip_dates_from_names(tree_path)
    if not dates:
        return "2026-06-01"
    return max(dates).strftime("%Y-%m-%d")


def load_tree(tree_path):
    tree = bt.loadNexus(
        tree_path,
        tip_regex=r"_(\d{4}-\d{2})(?:-|_)",
        date_fmt="%Y-%m",
        variableDate=False,
    )
    tree.setAbsoluteTime(bt.decimalDate(latest_tip_anchor(tree_path)))
    tree.traverse_tree()
    tree.sortBranches()
    return tree


def trait_map_value(node):
    return canon_state(getattr(node, "traits", {}).get(TRAIT_NAME))


def trait_prob_value(node):
    value = getattr(node, "traits", {}).get(f"{TRAIT_NAME}.prob")
    try:
        return float(value) if value is not None else 1.0
    except Exception:
        return 1.0


def draw_year_bands(ax, step_years=5, alpha=0.06):
    x0, x1 = ax.get_xlim()
    start = int(np.floor(x0 / step_years) * step_years)
    idx = 0
    for year in range(start, int(np.ceil(x1)) + step_years, step_years):
        if idx % 2 == 0:
            ax.axvspan(year, min(year + step_years, x1), color="k", alpha=alpha, zorder=0)
        idx += 1


def read_mj(cfg):
    path = os.path.join(BASE_DIR, cfg["mj"])
    df = pd.read_csv(path)
    df["startLocation"] = df["startLocation"].map(canon_state)
    df["endLocation"] = df["endLocation"].map(canon_state)
    df = df[df["startLocation"] != df["endLocation"]].copy()
    return df


def transition_counts_per_tree(df):
    tree_ids = sorted(df["treeId"].unique())
    g2e = (
        df[(df["startLocation"] == "Glo") & (df["endLocation"] == "EUR")]
        .groupby("treeId")
        .size()
        .reindex(tree_ids, fill_value=0)
    )
    e2g = (
        df[(df["startLocation"] == "EUR") & (df["endLocation"] == "Glo")]
        .groupby("treeId")
        .size()
        .reindex(tree_ids, fill_value=0)
    )
    counts = pd.DataFrame({
        "treeId": tree_ids,
        "Europe import": g2e.to_numpy(),
        "Global import": e2g.to_numpy(),
    })
    return counts


def plot_inset_hist(ax, df, hist_limits):
    counts = transition_counts_per_tree(df)["Europe import"]
    hpd = np.quantile(counts, [0.025, 0.975])
    mean_value = float(counts.mean())

    axins = inset_axes(
        ax,
        width="45%",
        height="40%",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.075, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    bins = hist_limits["bins"]
    weights = np.ones_like(counts, dtype=float) / len(counts)
    axins.hist(
        counts,
        bins=bins,
        weights=weights,
        color="#fdf0d5",
        edgecolor="#5f5f5f",
        linewidth=0.35,
    )
    axins.axvline(mean_value, color="black", linestyle="--", linewidth=0.75)
    axins.axvline(hpd[0], color="black", linestyle=":", linewidth=0.75)
    axins.axvline(hpd[1], color="black", linestyle=":", linewidth=0.75)
    axins.set_xlim(hist_limits["xlim"])
    axins.set_ylim(hist_limits["ylim"])
    axins.set_title("Posterior of global transition into Europe", fontsize=8.6, y=0.98, pad=-2.0)
    axins.set_xlabel("Global invasion dynamics into Europe", fontsize=8.0, labelpad=1.8)
    axins.set_ylabel("Proportion of trees", fontsize=8.0, labelpad=1.8)
    axins.tick_params(axis="both", labelsize=7.4, length=2.4, width=0.4)
    for spine in axins.spines.values():
        spine.set_linewidth(0.45)


def plot_tree(ax, cfg, mj_df, hist_limits):
    tree_path = os.path.join(BASE_DIR, cfg["tree"])
    print(f"Loading tree: {os.path.basename(tree_path)}")
    tree = load_tree(tree_path)

    uncertainty_color = "#B9B9B9"
    ramp_cache = {}

    def get_ramp(state):
        if state not in ramp_cache:
            base = STATE_COLORS.get(state, "#999999")
            ramp_cache[state] = clr.LinearSegmentedColormap.from_list(
                f"ramp_{state}", [uncertainty_color, base], N=256
            )
        return ramp_cache[state]

    branch_width = 0.82
    tip_size = 9

    for node in tree.Objects:
        x = getattr(node, "absoluteTime", None)
        parent = getattr(node, "parent", None)
        xp = getattr(parent, "absoluteTime", None)
        if x is None:
            x = 0.0
        if xp is None:
            xp = x
        y = node.y
        state = trait_map_value(node)
        prob = float(np.clip(trait_prob_value(node), 0.0, 1.0))
        color = get_ramp(state)(prob)
        node_branch_width = branch_width

        is_leaf = (getattr(node, "branchType", None) == "leaf") or node.__class__.__name__.lower() == "leaf"
        is_internal = (getattr(node, "branchType", None) == "node") or node.__class__.__name__.lower() == "node"

        if is_leaf:
            ax.scatter(x, y, s=tip_size * 1.25, facecolor="black", edgecolor="none", zorder=10)
            ax.scatter(x, y, s=tip_size, facecolor=color, edgecolor="none", zorder=11)
        elif is_internal:
            nchild = len(node.leaves) if hasattr(node, "leaves") else 1
            node_branch_width = branch_width + nchild * 0.0018
            if getattr(node, "children", None):
                ax.plot(
                    [x, x],
                    [node.children[-1].y, node.children[0].y],
                    lw=node_branch_width,
                    color=color,
                    zorder=9,
                )

        ax.plot([xp, x], [y, y], lw=node_branch_width, color=color, zorder=9)

    x_values = [getattr(k, "absoluteTime", np.nan) for k in tree.Objects]
    x_values = [v for v in x_values if np.isfinite(v)]
    x_min = max(1960, int(np.floor(min(x_values) / 5) * 5))
    x_max = max(2026.4, int(np.ceil(max(x_values))))
    ax.set_xlim(x_min, x_max)
    draw_year_bands(ax, step_years=5, alpha=0.06)

    ticks = list(range(int(np.floor(x_min / 5) * 5), int(np.ceil(x_max)) + 1, 5))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=10.6)
    ax.tick_params(axis="x", labelsize=10.6, size=3.5, width=0.85, color="#000000")
    ax.tick_params(axis="y", left=False, labelleft=False)
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    if hasattr(tree, "ySpan"):
        ax.set_ylim(-5, tree.ySpan + 18)

    ax.text(
        0.045,
        0.965,
        cfg["title"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        color="black",
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.18", fc=cfg["label_color"], ec="none"),
        zorder=120,
    )

    handles = [
        mlines.Line2D(
            [],
            [],
            color=STATE_COLORS[key],
            marker="o",
            markerfacecolor=STATE_COLORS[key],
            label=STATE_LABELS[key],
            markersize=9.4,
            linewidth=1.2,
        )
        for key in STATE_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.07, 0.13),
        frameon=False,
        markerfirst=True,
        prop={"size": 9.0},
    )

    plot_inset_hist(ax, mj_df, hist_limits)


def monthly_transition_table(df, x_start=2018.0, x_end=2026.25):
    d = df[(df["time"] >= x_start) & (df["time"] <= x_end)].copy()
    d["month"] = np.floor(d["time"] * 12) / 12
    months = np.arange(x_start, x_end + 1 / 12, 1 / 12)
    months = np.floor(months * 12) / 12
    tree_ids = sorted(df["treeId"].unique())

    records = []
    directions = [
        ("Europe import", "Glo", "EUR"),
        ("Global import", "EUR", "Glo"),
    ]
    for label, start, end in directions:
        counts = (
            d[(d["startLocation"] == start) & (d["endLocation"] == end)]
            .groupby(["treeId", "month"])
            .size()
            .rename("count")
        )
        full_index = pd.MultiIndex.from_product([tree_ids, months], names=["treeId", "month"])
        tmp = counts.reindex(full_index, fill_value=0).reset_index()
        tmp["direction"] = label
        records.append(tmp)

    return pd.concat(records, ignore_index=True)


def plot_temporal(ax, df, seed=42):
    tm = monthly_transition_table(df)
    x_start, x_end = 2018.0, 2026.25

    ax.axvspan(2018 + 1 / 12, 2020 + 2 / 12, color="#ededf7", zorder=0)
    ax.axvspan(2020 + 3 / 12, 2023 + 3 / 12, color="#fbe9e9", zorder=0)

    for direction in DIRECTION_ORDER:
        sub = tm[tm["direction"] == direction]
        monthly = (
            sub.groupby("month", observed=True)["count"]
            .agg(
                mean="mean",
                hpd_2_5=lambda x: np.quantile(x, 0.025),
                hpd_97_5=lambda x: np.quantile(x, 0.975),
            )
            .reset_index()
            .sort_values("month")
        )
        ax.fill_between(
            monthly["month"],
            monthly["hpd_2_5"],
            monthly["hpd_97_5"],
            color=DIRECTION_COLORS[direction],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            monthly["month"],
            monthly["mean"],
            color=DIRECTION_COLORS[direction],
            linewidth=1.55,
            label=direction,
            zorder=3,
        )

    ax.set_xlim(x_start, x_end)
    ax.set_ylabel("Monthly transition", fontsize=10.0)
    ax.tick_params(axis="both", labelsize=8.2, length=2.4, width=0.55)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    years = np.arange(2018, 2027)
    ticks = np.concatenate([years + 5 / 12, years + 11 / 12])
    ticks = ticks[(ticks >= x_start) & (ticks <= x_end)]
    labels = []
    for x in ticks:
        year = int(np.floor(x))
        month = int(round((x - year) * 12)) + 1
        labels.append(f"{'Jun' if month == 6 else 'Dec'}\n{year}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7.2)

    ymax = max(1.0, ax.get_ylim()[1])
    ax.text(2018.14, ymax * 0.95, "Measles resurgence\nperiod", fontsize=7.2, va="top")
    ax.text(2020.42, ymax * 0.95, "COVID-19 pandemic period", fontsize=7.2, va="top")
    ax.legend(frameon=False, fontsize=8.2, loc="upper right", handlelength=1.6)
    sns.despine(ax=ax)


def plot_violin(ax, df, violin_ylim):
    counts = transition_counts_per_tree(df)
    violin_df = counts.melt(id_vars="treeId", value_vars=DIRECTION_ORDER, var_name="direction", value_name="count")
    violin_df["direction"] = pd.Categorical(violin_df["direction"], DIRECTION_ORDER, ordered=True)
    violin_plot_df = violin_df.copy()
    trimmed_parts = []
    for direction, sub in violin_plot_df.groupby("direction", observed=True):
        low, high = sub["count"].quantile([0.025, 0.975])
        trimmed_parts.append(sub[(sub["count"] >= low) & (sub["count"] <= high)])
    violin_plot_df = pd.concat(trimmed_parts, ignore_index=True)

    sns.violinplot(
        data=violin_plot_df,
        x="direction",
        y="count",
        hue="direction",
        palette=DIRECTION_COLORS,
        dodge=False,
        inner=None,
        cut=0,
        linewidth=0.55,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    for i, direction in enumerate(DIRECTION_ORDER):
        vals = violin_df.loc[violin_df["direction"] == direction, "count"]
        hpd = np.quantile(vals, [0.025, 0.975])
        mean_value = vals.mean()
        ax.plot([i, i], [hpd[0], hpd[1]], color="white", linewidth=1.05, zorder=10)
        ax.scatter(i, mean_value, marker="D", facecolor="white", edgecolor="#4b4b4b", s=24, linewidth=0.65, zorder=11)

    ax.set_xlabel("")
    ax.set_ylabel("Delta transition per tree", fontsize=9.0)
    ax.set_ylim(violin_ylim)
    ax.tick_params(axis="x", labelsize=8.0, length=0)
    ax.tick_params(axis="y", labelsize=8.0, length=2.4, width=0.45)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)

    return violin_df


def summarize_study(cfg, mj_df):
    counts = transition_counts_per_tree(mj_df)
    rows = []
    for direction in DIRECTION_ORDER:
        vals = counts[direction].to_numpy()
        hpd = np.quantile(vals, [0.025, 0.975])
        rows.append({
            "study": cfg["id"],
            "direction": direction,
            "n_trees": len(vals),
            "mean": vals.mean(),
            "median": np.median(vals),
            "hpd_2_5": hpd[0],
            "hpd_97_5": hpd[1],
        })
    return pd.DataFrame(rows)


def compute_plot_limits(mj_by_study):
    europe_counts = []
    all_transition_counts = []
    for mj_df in mj_by_study.values():
        counts = transition_counts_per_tree(mj_df)
        europe_counts.append(counts["Europe import"].to_numpy())
        for direction in DIRECTION_ORDER:
            all_transition_counts.append(counts[direction].to_numpy())

    hist_values = np.concatenate(europe_counts)
    hist_hpd = np.quantile(hist_values, [0.025, 0.975])
    hist_xlim = (np.floor(hist_hpd[0] - 6), np.ceil(hist_hpd[1] + 6))
    hist_bins = np.arange(hist_xlim[0] - 0.5, hist_xlim[1] + 1.5, 1)
    ymax = 0
    for values in europe_counts:
        weights = np.ones_like(values, dtype=float) / len(values)
        hist, _ = np.histogram(values, bins=hist_bins, weights=weights)
        ymax = max(ymax, float(hist.max()))

    all_counts = np.concatenate(all_transition_counts)
    violin_hpd = np.quantile(all_counts, [0.025, 0.975])

    return {
        "hist": {
            "xlim": hist_xlim,
            "ylim": (0, np.ceil((ymax * 1.12) * 100) / 100),
            "bins": hist_bins,
        },
        "violin_ylim": (0, np.ceil(violin_hpd[1] + 6)),
    }


def main():
    for cfg in STUDIES:
        for key in ["tree", "mj"]:
            path = os.path.join(BASE_DIR, cfg[key])
            if not os.path.exists(path):
                raise FileNotFoundError(path)

    mj_by_study = {}
    summaries = []
    for cfg in STUDIES:
        print(f"Reading MJ: {cfg['mj']}")
        mj = read_mj(cfg)
        mj_by_study[cfg["id"]] = mj
        summaries.append(summarize_study(cfg, mj))

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(os.path.join(OUT_DIR, "dta2_mj_transition_summary_2026.csv"), index=False)
    plot_limits = compute_plot_limits(mj_by_study)

    fig = plt.figure(figsize=(22, 9.2), facecolor="white")
    outer = fig.add_gridspec(1, 3, wspace=0.07)

    all_violin = []
    for col, cfg in enumerate(STUDIES):
        sub = outer[col].subgridspec(2, 1, height_ratios=[3.25, 1.35], hspace=0.055)
        ax_tree = fig.add_subplot(sub[0, 0])
        lower = sub[1, 0].subgridspec(1, 2, width_ratios=[3.1, 0.92], wspace=0.12)
        ax_temporal = fig.add_subplot(lower[0, 0])
        ax_violin = fig.add_subplot(lower[0, 1])

        mj = mj_by_study[cfg["id"]]
        plot_tree(ax_tree, cfg, mj, plot_limits["hist"])
        plot_temporal(ax_temporal, mj, seed=42 + col)
        violin_df = plot_violin(ax_violin, mj, plot_limits["violin_ylim"])
        violin_df["study"] = cfg["id"]
        all_violin.append(violin_df)

        ax_tree.text(
            -0.04,
            1.025,
            cfg["panel"],
            transform=ax_tree.transAxes,
            ha="left",
            va="top",
            fontsize=17,
            color="black",
            fontweight="normal",
        )

    pd.concat(all_violin, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "dta2_mj_transition_counts_per_tree_2026.csv"),
        index=False,
    )

    fig.subplots_adjust(left=0.025, right=0.992, top=0.985, bottom=0.06)
    pdf_path = os.path.join(OUT_DIR, "Euro_resurgence_2dta_phylogeny_invasion_MJ_temporal_2026.pdf")
    png_path = os.path.join(OUT_DIR, "Euro_resurgence_2dta_phylogeny_invasion_MJ_temporal_2026.png")
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=260)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
