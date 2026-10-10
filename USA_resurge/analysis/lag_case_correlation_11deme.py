from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import core_dta_visualization as core


ROOT = Path(__file__).resolve().parent
OUT = ROOT
SOURCE = ROOT / "N450"
FIG = OUT / "figures" / "lag_test"
TAB = OUT / "tables" / "lag_test"
for directory in (FIG, TAB):
    directory.mkdir(parents=True, exist_ok=True)

GLOBAL_MJ = SOURCE / "USA_cap3_monthly_MJ_global_to_USA_imports_summary.csv"
GLOBAL_CASES = SOURCE / "USA_cap3_WHO_global_non_USA_monthly_measles_cases_since2018.csv"
USA_CASES = SOURCE / "USA_cap3_WHO_USA_monthly_measles_cases_since2018.csv"
REGIONAL_MJ = core.REGIONAL["mj"]

REGIONAL_START = pd.Timestamp("2023-07-01")
REGIONAL_CUTOFF = pd.Timestamp("2026-06-01")
LAGS = np.arange(-6, 7)

C_GLOBAL = "#E07B54"
C_USA = "#6A4C93"
C_GLOBAL_MJ = "#4A7BA4"
C_REGIONAL_MJ = "#2F7F79"
C_INK = "#252A2E"
C_GRID = "#DDE1E3"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.labelsize": 14.5,
    "axes.titlesize": 15.5,
    "xtick.labelsize": 13.0,
    "ytick.labelsize": 13.0,
    "legend.fontsize": 12.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def decimal_date(value: pd.Timestamp) -> float:
    dt = value.to_pydatetime()
    start = datetime(dt.year, 1, 1)
    end = datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - start).total_seconds() / (end - start).total_seconds()


def load_global_data() -> pd.DataFrame:
    mj = pd.read_csv(GLOBAL_MJ)
    global_cases = pd.read_csv(GLOBAL_CASES, parse_dates=["date"])
    usa_cases = pd.read_csv(USA_CASES, parse_dates=["date"])
    expected_dates = pd.date_range("2018-01-01", periods=len(mj), freq="MS")
    expected_decimals = np.array([decimal_date(date) for date in expected_dates])
    if not np.allclose(mj["month_dec"].to_numpy(float), expected_decimals, atol=1e-8):
        raise ValueError("Global MJ month_dec values are not a contiguous monthly series from 2018-01")
    mj = mj.assign(date=expected_dates)
    merged = (
        mj.merge(global_cases[["date", "global_non_usa_monthly_cases"]], on="date", validate="one_to_one")
        .merge(usa_cases[["date", "usa_monthly_cases"]], on="date", validate="one_to_one")
        .dropna()
        .sort_values("date")
    )
    if merged.empty:
        raise ValueError("No overlapping global MJ and case months")
    return merged


def summarize_regional_mj() -> pd.DataFrame:
    events = pd.read_csv(REGIONAL_MJ)
    required = {"treeId", "startLocation", "endLocation", "time"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing regional MJ columns: {sorted(missing)}")

    trees = np.sort(events["treeId"].dropna().unique())
    edges = pd.date_range(REGIONAL_START, REGIONAL_CUTOFF + pd.DateOffset(months=1), freq="MS")
    edge_decimals = np.array([decimal_date(date) for date in edges])
    domestic = events.loc[
        events["startLocation"].ne("nonUSA")
        & events["endLocation"].ne("nonUSA")
        & events["startLocation"].ne(events["endLocation"])
        & events["time"].ge(edge_decimals[0])
        & events["time"].lt(edge_decimals[-1])
    ]

    rows: list[dict[str, object]] = []
    for tree_id in trees:
        values = domestic.loc[domestic["treeId"].eq(tree_id), "time"].to_numpy(float)
        counts, _ = np.histogram(values, bins=edge_decimals)
        rows.extend({"treeId": tree_id, "date": date, "count": int(count)}
                    for date, count in zip(edges[:-1], counts))
    by_tree = pd.DataFrame(rows)
    summary = (
        by_tree.groupby("date", as_index=False)["count"]
        .agg(
            mean="mean",
            q025=lambda values: np.quantile(values, .025),
            q975=lambda values: np.quantile(values, .975),
        )
    )
    by_tree.to_csv(TAB / "interregion_monthly_MJ_counts_by_tree.csv", index=False)
    summary.to_csv(TAB / "interregion_monthly_MJ_counts_summary.csv", index=False)
    return summary


def merge_regional_cases(regional: pd.DataFrame) -> pd.DataFrame:
    cases = pd.read_csv(USA_CASES, parse_dates=["date"])
    merged = regional.merge(
        cases[["date", "usa_monthly_cases"]], on="date", how="inner", validate="one_to_one",
    ).dropna().sort_values("date")
    if len(merged) < 12:
        raise ValueError(f"Only {len(merged)} overlapping regional MJ/case months")
    merged.to_csv(TAB / "interregion_monthly_MJ_USA_cases_overlap.csv", index=False)
    return merged


def lag_correlations(cases: np.ndarray, mj: np.ndarray, comparison: str) -> pd.DataFrame:
    if len(cases) != len(mj):
        raise ValueError("Lag-correlation inputs must have the same length")
    rows = []
    for lag in LAGS:
        if lag < 0:
            x, y = cases[-lag:], mj[:lag]
        elif lag > 0:
            x, y = cases[:-lag], mj[lag:]
        else:
            x, y = cases, mj
        if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
            r_value, p_value = np.nan, np.nan
        else:
            result = spearmanr(x, y)
            r_value, p_value = float(result.statistic), float(result.pvalue)
        rows.append({
            "comparison": comparison,
            "lag_months": int(lag),
            "spearman_rho": r_value,
            "p_value": p_value,
            "n_months": len(x),
        })
    return pd.DataFrame(rows)


def panel_label(ax: plt.Axes, label: str, x: float = -.055) -> None:
    ax.text(x, 1.055, label, transform=ax.transAxes, fontsize=20,
            fontweight="bold", ha="left", va="bottom", color=C_INK)


def style_temporal_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=C_GRID, lw=.65, alpha=.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#62686C")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", length=3.5, color="#62686C")


def plot_scatter_panel(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str,
                       title: str, xlabel: str, ylabel: str, label: str) -> None:
    if len(x) != len(y) or len(x) < 4:
        raise ValueError("Regression scatter inputs must contain at least four paired values")
    rho, p_value = spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    fit_y = slope * fit_x + intercept

    ax.scatter(x, y, s=46, color=C_INK, alpha=.40, edgecolor="none", zorder=3)
    ax.scatter(x, y, s=29, color=color, alpha=.72, edgecolor="none", zorder=4)
    ax.plot(fit_x, fit_y, color=color, lw=2.0, ls=(0, (5, 3)), zorder=5)
    p_text = "p < 0.001" if p_value < .001 else f"p = {p_value:.3f}"
    ax.text(.98, .95, f"Spearman r = {rho:.3f}\n{p_text}",
            transform=ax.transAxes, ha="right", va="top", fontsize=12.0, color=C_INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94, "pad": 1.8},
            zorder=7)
    ax.set_title(title, loc="left", fontsize=14.0, pad=10)
    ax.set_xlabel(xlabel, fontsize=12.5)
    ax.set_ylabel(ylabel, fontsize=13.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(color=C_GRID, lw=.6, alpha=.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, label, x=-.10)


def plot_lag_panel(ax: plt.Axes, stats: pd.DataFrame, color: str,
                   title: str, label: str, common_limit: float) -> None:
    stats = stats.sort_values("lag_months")
    lags = stats["lag_months"].to_numpy(int)
    values = stats["spearman_rho"].to_numpy(float)
    ax.bar(lags, values, width=.72, color=color, alpha=.90,
           edgecolor="white", linewidth=.5, zorder=3)
    ax.axhline(0, color=C_INK, lw=.8)
    ax.axvline(0, color="#777D81", lw=.8, ls=(0, (3, 2)))
    finite = stats.dropna(subset=["spearman_rho"])
    peak = finite.loc[finite["spearman_rho"].abs().idxmax()]
    p_text = "p < 0.001" if peak.p_value < .001 else f"p = {peak.p_value:.3f}"
    ax.text(.98, .94, f"Spearman r = {peak.spearman_rho:.3f}\n{p_text}",
            transform=ax.transAxes, ha="right", va="top", fontsize=12.0, color=C_INK)
    ax.set_title(title, loc="left", fontsize=14.0, pad=10)
    ax.set_xlabel("Lag (months; positive = cases lead MJ)", fontsize=12.5)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_ylim(0, common_limit)
    ax.grid(axis="y", color=C_GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, label, x=-.10)


def plot_figure(global_data: pd.DataFrame, regional_data: pd.DataFrame,
                lag_sets: list[pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(17.2, 18.2), facecolor="white")
    grid = fig.add_gridspec(
        4, 3, height_ratios=[1.08, 1.02, .84, .78], hspace=.56, wspace=.34,
    )

    ax_a = fig.add_subplot(grid[0, :])
    ax_a_usa = ax_a.twinx()
    ax_a_mj = ax_a.twinx()
    ax_a_mj.spines["right"].set_position(("axes", 1.065))
    dates = global_data["date"]
    global_cases = global_data["global_non_usa_monthly_cases"].to_numpy(float) / 1e4
    usa_cases = global_data["usa_monthly_cases"].to_numpy(float)
    global_mj = global_data["mean"].to_numpy(float)
    ax_a.fill_between(dates, global_cases, color=C_GLOBAL, alpha=.16, linewidth=0)
    ax_a.plot(dates, global_cases, color=C_GLOBAL, lw=1.35)
    ax_a_usa.fill_between(dates, usa_cases, color=C_USA, alpha=.12, linewidth=0)
    ax_a_usa.plot(dates, usa_cases, color=C_USA, lw=1.35)
    ax_a_mj.fill_between(dates, global_data["q025"].to_numpy(float),
                         global_data["q975"].to_numpy(float),
                         color=C_GLOBAL_MJ, alpha=.18, linewidth=0)
    ax_a_mj.plot(dates, global_mj, color=C_GLOBAL_MJ, lw=2.0)
    ax_a.set_ylabel("Global non-USA cases (x10^4)", color=C_GLOBAL)
    ax_a_usa.set_ylabel("USA monthly cases", color=C_USA)
    ax_a_mj.set_ylabel("Global-to-USA MJ imports/tree", color=C_GLOBAL_MJ)
    ax_a.tick_params(axis="y", colors=C_GLOBAL)
    ax_a_usa.tick_params(axis="y", colors=C_USA)
    ax_a_mj.tick_params(axis="y", colors=C_GLOBAL_MJ)
    ax_a.set_xlim(dates.min(), dates.max())
    style_temporal_axis(ax_a)
    ax_a.set_title("Global-to-USA introductions and measles incidence", loc="left", pad=8)
    handles_a = [
        mpl.lines.Line2D([], [], color=C_GLOBAL, lw=2, label="Global non-USA cases (x10^4)"),
        mpl.lines.Line2D([], [], color=C_USA, lw=2, label="USA monthly cases"),
        mpl.lines.Line2D([], [], color=C_GLOBAL_MJ, lw=2, label="Global-to-USA MJ mean"),
        mpl.patches.Patch(facecolor=C_GLOBAL_MJ, alpha=.18, edgecolor="none",
                          label="MJ 95% posterior interval"),
    ]
    ax_a.legend(handles=handles_a, frameon=False, loc="upper left", ncol=2)
    panel_label(ax_a, "A")

    ax_b = fig.add_subplot(grid[1, :])
    ax_b_mj = ax_b.twinx()
    dates_b = regional_data["date"]
    cases_b = regional_data["usa_monthly_cases"].to_numpy(float)
    mj_b = regional_data["mean"].to_numpy(float)
    ax_b.fill_between(dates_b, cases_b, color=C_USA, alpha=.13, linewidth=0)
    ax_b.plot(dates_b, cases_b, color=C_USA, lw=1.55)
    ax_b_mj.fill_between(dates_b, regional_data["q025"].to_numpy(float),
                         regional_data["q975"].to_numpy(float),
                         color=C_REGIONAL_MJ, alpha=.18, linewidth=0)
    ax_b_mj.plot(dates_b, mj_b, color=C_REGIONAL_MJ, lw=2.1)
    ax_b.set_ylabel("USA monthly cases", color=C_USA)
    ax_b_mj.set_ylabel("Interregional MJ/tree", color=C_REGIONAL_MJ)
    ax_b.tick_params(axis="y", colors=C_USA)
    ax_b_mj.tick_params(axis="y", colors=C_REGIONAL_MJ)
    ax_b.set_xlim(dates_b.min(), dates_b.max())
    style_temporal_axis(ax_b)
    ax_b.set_title("11-deme interregional dynamics and USA measles incidence", loc="left", pad=8)
    handles_b = [
        mpl.lines.Line2D([], [], color=C_USA, lw=2, label="USA monthly cases"),
        mpl.lines.Line2D([], [], color=C_REGIONAL_MJ, lw=2, label="Interregional MJ mean"),
        mpl.patches.Patch(facecolor=C_REGIONAL_MJ, alpha=.18, edgecolor="none",
                          label="MJ 95% posterior interval"),
    ]
    ax_b.legend(handles=handles_b, frameon=False, loc="upper left", ncol=3)
    panel_label(ax_b, "B")

    scatter_specs = [
        (
            global_data["global_non_usa_monthly_cases"].to_numpy(float) / 1e4,
            global_data["mean"].to_numpy(float), C_GLOBAL,
            "Global cases vs global-to-USA MJ",
            "Global non-USA monthly cases (x10^4)",
            "Global-to-USA MJ imports/tree (mean)", "C",
        ),
        (
            global_data["usa_monthly_cases"].to_numpy(float),
            global_data["mean"].to_numpy(float), C_USA,
            "USA cases vs global-to-USA MJ",
            "USA monthly cases", "Global-to-USA MJ imports/tree (mean)", "D",
        ),
        (
            regional_data["usa_monthly_cases"].to_numpy(float),
            regional_data["mean"].to_numpy(float), C_REGIONAL_MJ,
            "USA cases vs interregional MJ",
            "USA monthly cases", "Interregional MJ/tree (mean)", "E",
        ),
    ]
    scatter_axes = [fig.add_subplot(grid[2, index]) for index in range(3)]
    for index, (ax, spec) in enumerate(zip(scatter_axes, scatter_specs)):
        plot_scatter_panel(ax, *spec)
        if index > 0:
            ax.set_ylabel("")

    common_limit = min(1.0, max(.3, np.ceil(max(
        stats["spearman_rho"].max() for stats in lag_sets
    ) * 10) / 10 + .1))
    lag_titles = [
        "Global cases vs global-to-USA MJ",
        "USA cases vs global-to-USA MJ",
        "USA cases vs interregional MJ",
    ]
    lag_colors = [C_GLOBAL, C_USA, C_REGIONAL_MJ]
    lag_labels = ["F", "G", "H"]
    lag_axes = [fig.add_subplot(grid[3, index]) for index in range(3)]
    for index, (ax, stats, color, title, label) in enumerate(zip(
            lag_axes, lag_sets, lag_colors, lag_titles, lag_labels)):
        plot_lag_panel(ax, stats, color, title, label, common_limit)
        if index == 0:
            ax.set_ylabel("Lagged Spearman correlation (r)", fontsize=13.5)
        else:
            ax.set_yticklabels([])

    fig.suptitle("Temporal and lagged associations between Markov jumps and measles incidence",
                 x=.07, ha="left", y=.994, fontsize=22, fontweight="bold", color=C_INK)
    fig.text(.07, .968,
             "Monthly posterior mean MJ counts; ribbons show 95% posterior intervals; scatter panels pair contemporaneous months; dashed lines show linear fits",
             ha="left", fontsize=14.0, color="#596064")
    fig.subplots_adjust(left=.09, right=.90, top=.94, bottom=.055)
    stem = "USA_MJ_case_correlation"
    fig.savefig(FIG / f"{stem}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    global_data = load_global_data()
    regional_summary = summarize_regional_mj()
    regional_data = merge_regional_cases(regional_summary)

    lag_global_cases = lag_correlations(
        global_data["global_non_usa_monthly_cases"].to_numpy(float),
        global_data["mean"].to_numpy(float),
        "Global non-USA cases vs global-to-USA MJ",
    )
    lag_usa_global_mj = lag_correlations(
        global_data["usa_monthly_cases"].to_numpy(float),
        global_data["mean"].to_numpy(float),
        "USA cases vs global-to-USA MJ",
    )
    lag_usa_regional_mj = lag_correlations(
        regional_data["usa_monthly_cases"].to_numpy(float),
        regional_data["mean"].to_numpy(float),
        "USA cases vs interregional MJ",
    )
    lag_sets = [lag_global_cases, lag_usa_global_mj, lag_usa_regional_mj]
    pd.concat(lag_sets, ignore_index=True).to_csv(TAB / "USA_MJ_case_lag_correlations.csv", index=False)
    plot_figure(global_data, regional_data, lag_sets)
    print(f"Global overlap: {len(global_data)} months")
    print(f"Regional overlap: {len(regional_data)} months")
    for stats in lag_sets:
        peak = stats.loc[stats.spearman_rho.abs().idxmax()]
        print(f"{peak.comparison}: peak lag {int(peak.lag_months):+d}, rho={peak.spearman_rho:.3f}, p={peak.p_value:.4g}, n={int(peak.n_months)}")


if __name__ == "__main__":
    main()
