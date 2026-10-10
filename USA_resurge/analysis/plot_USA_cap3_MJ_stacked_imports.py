"""
Plot monthly Markov-jump source-specific introductions into USA.

Definition:
For each posterior tree, count Markov jumps with endLocation == USA and
startLocation != USA in monthly bins from 2018-01-01 to 2026-06-01. The plotted
value is the posterior mean count per tree per month.
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import visualize_USA_cap3_final_figures as base


OUT = base.OUT
MJ = base.MJ
START = pd.Timestamp("2018-01-01")
CUTOFF = pd.Timestamp("2026-06-01")
TOP_N = 10

PERIODS = [
    ("Measles resurgence", pd.Timestamp("2018-02-01"), pd.Timestamp("2020-03-31"), "#7E80C9", 0.13),
    ("COVID-19 pandemic", pd.Timestamp("2020-04-01"), pd.Timestamp("2023-04-30"), "#D66B6B", 0.13),
]
DONUT_PERIODS = [
    ("Measles resurgence\nFeb 2018-Mar 2020", pd.Timestamp("2018-02-01"), pd.Timestamp("2020-03-31"), "#7E80C9"),
    ("COVID-19 pandemic\nApr 2020-Apr 2023", pd.Timestamp("2020-04-01"), pd.Timestamp("2023-04-30"), "#D66B6B"),
    ("Post-COVID-19\nMay 2023-Jun 2026", pd.Timestamp("2023-05-01"), CUTOFF, "#69C6C4"),
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei", "SimSun"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def decimal_date(value) -> float:
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = pd.Timestamp(value).to_pydatetime()
    start = datetime(dt.year, 1, 1)
    end = datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - start).total_seconds() / (end - start).total_seconds()


def month_edges() -> pd.DatetimeIndex:
    return pd.date_range(START, CUTOFF + pd.DateOffset(months=1), freq="MS")


def source_label(x: str) -> str:
    return str(x).replace("_", " ")


def compute_monthly_source_means() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(MJ)
    trees = sorted(df["treeId"].dropna().unique())
    edges = month_edges()
    edge_dec = np.array([decimal_date(x) for x in edges])

    imp = df.loc[
        (df["time"] >= edge_dec[0])
        & (df["time"] < edge_dec[-1])
        & df["endLocation"].eq("USA")
        & df["startLocation"].ne("USA")
    ].copy()
    imp["source"] = imp["startLocation"].astype(str)

    total_by_source = (
        imp.groupby("source").size().sort_values(ascending=False) / max(len(trees), 1)
    )
    top_sources = total_by_source.head(TOP_N).index.tolist()
    imp["source_group"] = np.where(imp["source"].isin(top_sources), imp["source"], "Other_sources")

    rows = []
    for tree_id in trees:
        tdf = imp.loc[imp["treeId"].eq(tree_id)]
        for src in top_sources + ["Other_sources"]:
            vals = tdf.loc[tdf["source_group"].eq(src), "time"].to_numpy()
            counts, _ = np.histogram(vals, bins=edge_dec)
            for month, left_dec, count in zip(edges[:-1], edge_dec[:-1], counts):
                rows.append({
                    "treeId": tree_id,
                    "month": month.strftime("%Y-%m-%d"),
                    "month_dec": left_dec,
                    "source": src,
                    "imports": int(count),
                })
    by_tree = pd.DataFrame(rows)
    summary = (
        by_tree.groupby(["month", "month_dec", "source"], as_index=False)["imports"]
        .agg(mean_imports_per_tree="mean",
             q025=lambda x: np.quantile(x, 0.025),
             q975=lambda x: np.quantile(x, 0.975))
    )
    totals = (
        by_tree.groupby(["month", "month_dec", "treeId"], as_index=False)["imports"].sum()
        .groupby(["month", "month_dec"], as_index=False)["imports"]
        .agg(total_mean_imports_per_tree="mean",
             total_q025=lambda x: np.quantile(x, 0.025),
             total_q975=lambda x: np.quantile(x, 0.975))
    )
    by_tree.to_csv(os.path.join(OUT, "USA_cap3_monthly_MJ_imports_to_USA_by_source_by_tree.csv"), index=False)
    summary.to_csv(os.path.join(OUT, "USA_cap3_monthly_MJ_imports_to_USA_by_source_summary.csv"), index=False)
    totals.to_csv(os.path.join(OUT, "USA_cap3_monthly_MJ_imports_to_USA_total_summary.csv"), index=False)
    return summary, totals


def smooth_monthly_values(summary: pd.DataFrame, totals: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if window <= 1:
        return summary.copy(), totals.copy()
    smoothed = summary.sort_values(["source", "month_dec"]).copy()
    smoothed["mean_imports_per_tree"] = (
        smoothed.groupby("source")["mean_imports_per_tree"]
        .transform(lambda x: x.rolling(window=window, center=True, min_periods=1).mean())
    )
    smoothed["q025"] = (
        smoothed.groupby("source")["q025"]
        .transform(lambda x: x.rolling(window=window, center=True, min_periods=1).mean())
    )
    smoothed["q975"] = (
        smoothed.groupby("source")["q975"]
        .transform(lambda x: x.rolling(window=window, center=True, min_periods=1).mean())
    )

    total_smoothed = totals.sort_values("month_dec").copy()
    for col in ["total_mean_imports_per_tree", "total_q025", "total_q975"]:
        total_smoothed[col] = total_smoothed[col].rolling(window=window, center=True, min_periods=1).mean()
    return smoothed, total_smoothed


def plot_stacked(summary: pd.DataFrame, totals: pd.DataFrame, stem_suffix: str = "") -> None:
    pivot = (
        summary.pivot_table(index=["month", "month_dec"], columns="source",
                            values="mean_imports_per_tree", aggfunc="sum", fill_value=0)
        .reset_index()
        .sort_values("month_dec")
    )
    source_cols = [c for c in pivot.columns if c not in {"month", "month_dec"}]
    source_cols = sorted(
        source_cols,
        key=lambda c: pivot[c].sum(),
        reverse=True,
    )
    if "Other_sources" in source_cols:
        source_cols = [c for c in source_cols if c != "Other_sources"] + ["Other_sources"]

    colors = [
        base.STATE_COLORS.get(src, "#B8BDC7") if src != "Other_sources" else "#C7CBD1"
        for src in source_cols
    ]

    x = pivot["month_dec"].to_numpy()
    y = [pivot[src].to_numpy(dtype=float) for src in source_cols]
    total = totals.sort_values("month_dec")

    fig = plt.figure(figsize=(12.0, 7.8), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.25], hspace=0.35, wspace=0.26)
    ax = fig.add_subplot(gs[0, :])
    for label, start, end, color, alpha in PERIODS:
        ax.axvspan(decimal_date(start), decimal_date(end), color=color, alpha=alpha, zorder=0)

    ax.stackplot(
        x,
        y,
        colors=colors,
        labels=[source_label(s) for s in source_cols],
        alpha=0.88,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        total["month_dec"],
        total["total_mean_imports_per_tree"],
        color="#111827",
        lw=1.9,
        label="Total mean imports",
        zorder=5,
    )
    ax.fill_between(
        total["month_dec"].to_numpy(dtype=float),
        total["total_q025"].to_numpy(dtype=float),
        total["total_q975"].to_numpy(dtype=float),
        color="#111827",
        alpha=0.10,
        lw=0,
        zorder=1,
    )

    xticks = np.arange(2018, 2027, 1)
    ax.set_xlim(decimal_date(START), decimal_date(CUTOFF))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=14.3)
    ax.set_ylabel("Mean monthly MJ imports into USA\nper posterior tree", fontsize=15.6)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=14.3, length=3)
    ax.grid(axis="y", color="#D6DAE2", lw=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#4B5563")

    total_handle = mpl.lines.Line2D([], [], color="#111827", lw=1.9, label="Total mean imports")
    interval_handle = mpl.patches.Patch(facecolor="#111827", alpha=0.10, edgecolor="none",
                                        label="Total 95% posterior interval")
    period_handles = [
        mpl.patches.Patch(facecolor=color, alpha=alpha, edgecolor="none", label=label)
        for label, _, _, color, alpha in PERIODS
    ]
    leg1 = ax.legend(
        [total_handle, interval_handle] + period_handles,
        ["Total mean imports", "Total 95% posterior interval"] + [h.get_label() for h in period_handles],
        loc="upper left",
        frameon=False,
        fontsize=10,
        ncol=2,
        bbox_to_anchor=(0.0, 1.02),
    )
    ax.add_artist(leg1)
    source_handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="none", label=source_label(src))
        for src, color in zip(source_cols, colors)
    ]
    source_legend = fig.legend(
        source_handles,
        [h.get_label() for h in source_handles],
        loc="center left",
        bbox_to_anchor=(0.815, 0.47),
        frameon=False,
        fontsize=9.8,
        title="Source region",
        title_fontsize=11.0,
    )

    donut_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    donut_payload = []
    for period_label, start, end, period_color in DONUT_PERIODS:
        start_dec = decimal_date(start)
        end_dec = decimal_date(end + pd.DateOffset(months=1))
        period = pivot[(pivot["month_dec"] >= start_dec) & (pivot["month_dec"] < end_dec)].copy()
        values = period[source_cols].sum(axis=0).to_numpy(dtype=float)
        keep = values > 0
        vals = values[keep]
        colors_period = [colors[i] for i, ok in enumerate(keep) if ok]
        total = float(vals.sum())
        donut_payload.append({
            "label": period_label,
            "start": start,
            "end": end,
            "start_dec": start_dec,
            "end_dec": end_dec,
            "period_color": period_color,
            "vals": vals,
            "colors": colors_period,
            "total": total,
        })

    donut_totals = np.array([d["total"] for d in donut_payload], dtype=float)
    tmin, tmax = float(np.nanmin(donut_totals)), float(np.nanmax(donut_totals))
    denom = max(tmax - tmin, 1e-9)

    for dax, info in zip(donut_axes, donut_payload):
        vals = info["vals"]
        total = info["total"]
        radius = 0.82 + 0.14 * ((total - tmin) / denom)
        if total > 0:
            dax.pie(
                vals,
                colors=info["colors"],
                startangle=90,
                counterclock=False,
                radius=radius,
                wedgeprops=dict(width=radius * 0.36, edgecolor="white", linewidth=0.8),
            )
            dax.text(0, 0.08, f"{int(round(total))}", ha="center", va="center", fontsize=13, fontweight="bold")
            dax.text(0, -0.13, "MJ/tree", ha="center", va="center", fontsize=8.8, color="#4B5563")
        dax.set_title(info["label"], fontsize=10.5, pad=4)
        dax.set_aspect("equal")
        dax.axis("off")

    fig.subplots_adjust(left=0.10, right=0.78, top=0.96, bottom=0.07)
    stem = f"USA_cap3_MJ_stacked_imports_to_USA_by_source_since2018{stem_suffix}"
    for ext, dpi in [("png", 360), ("pdf", 300), ("svg", 300)]:
        out = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main() -> None:
    summary, totals = compute_monthly_source_means()
    plot_stacked(summary, totals)



if __name__ == "__main__":
    main()
