from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import core_dta_visualization as core


BASE = Path(__file__).resolve().parent / "WGS"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures" / "standalone_phylogeny_full_dates"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "legend.fontsize": 9.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def save_figure(fig: plt.Figure, stem: str, tight: bool = True) -> None:
    bbox_inches = "tight" if tight else None
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches=bbox_inches, facecolor="white")
    fig.savefig(FIG / f"{stem}.png", dpi=340, bbox_inches=bbox_inches, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}", flush=True)


def make_date_ticks(
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    labelled_months: tuple[int, ...],
) -> tuple[list[float], list[str], list[float]]:
    first_date = pd.Timestamp(first_date).normalize()
    last_date = pd.Timestamp(last_date).normalize()
    monthly = pd.date_range(
        pd.Timestamp(first_date.year, 1, 1),
        pd.Timestamp(last_date.year, 12, 1),
        freq="MS",
    )
    boundary_gap = pd.Timedelta(days=45)
    labelled = [
        d for d in monthly
        if d.month in labelled_months
        and d > first_date + boundary_gap
        and d < last_date - boundary_gap
    ]
    major_dates = [first_date, *labelled, last_date]
    major_labels = [
        first_date.strftime("%Y-%m-%d"),
        *[d.strftime("%Y-%m") for d in labelled],
        last_date.strftime("%Y-%m-%d"),
    ]
    major_positions = [core.decimal_year(d) for d in major_dates]
    minor_positions = [
        core.decimal_year(d) for d in monthly
        if first_date < d < last_date and d not in labelled
    ]
    return major_positions, major_labels, minor_positions


def decimal_to_date(value: float) -> pd.Timestamp:
    year = int(math.floor(value))
    days = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365
    offset = int(round((value - year) * days))
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=offset)


def make_full_scale_ticks(
    root_decimal: float,
    last_date: pd.Timestamp,
    mode: str,
) -> tuple[list[float], list[str], list[float]]:
    root_date = decimal_to_date(root_decimal)
    last_date = pd.Timestamp(last_date).normalize()
    last_decimal = core.decimal_year(last_date)
    if mode == "decade":
        first_decade = int(math.ceil(root_decimal / 10) * 10)
        decades = list(range(first_decade, int(last_decimal) + 1, 10))
        decades = [year for year in decades if root_decimal + 1 < year < last_decimal - 1]
        major_positions = [root_decimal, *[float(year) for year in decades], last_decimal]
        major_labels = [
            f"~{root_date:%Y-%m-%d}",
            *[str(year) for year in decades],
            last_date.strftime("%Y-%m-%d"),
        ]
        minor_positions = [
            float(year) for year in range(int(math.ceil(root_decimal / 5) * 5), int(last_decimal), 5)
            if year not in decades
        ]
        return major_positions, major_labels, minor_positions
    if mode == "halfyear":
        major_positions, major_labels, minor_positions = make_date_ticks(
            root_date, last_date, (1, 7)
        )
        major_positions[0] = root_decimal
        major_labels[0] = f"~{root_date:%Y-%m-%d}"
        return major_positions, major_labels, minor_positions
    if mode == "year":
        intermediate_years = list(range(root_date.year + 1, last_date.year))
        major_positions = [root_decimal, *[float(year) for year in intermediate_years], last_decimal]
        major_labels = [f"~{root_date.year}", *[str(year) for year in intermediate_years], str(last_date.year)]
        minor_positions = [
            core.decimal_year(pd.Timestamp(year, 7, 1))
            for year in range(root_date.year, last_date.year + 1)
            if root_decimal < core.decimal_year(pd.Timestamp(year, 7, 1)) < last_decimal
        ]
        return major_positions, major_labels, minor_positions
    raise ValueError(f"Unsupported tick mode: {mode}")


def draw_calendar_background(
    ax: plt.Axes,
    first_decimal: float,
    last_decimal: float,
    band_years: int,
) -> None:
    first_band = int(math.floor(first_decimal / band_years) * band_years)
    for left_year in range(first_band, math.ceil(last_decimal) + band_years, band_years):
        band_number = left_year // band_years
        if band_number % 2 == 0:
            left = max(first_decimal, float(left_year))
            right = min(last_decimal, float(left_year + band_years))
            if right > left:
                ax.axvspan(left, right, color="#EFEFEF", alpha=.78, zorder=0)


def draw_standalone_tree(
    prepared,
    colors: dict[str, str],
    legend_order: list[str],
    first_sample_date: pd.Timestamp,
    last_date: pd.Timestamp,
    tick_mode: str,
    band_years: int,
    title: str,
    subtitle_label: str,
    figure_size: tuple[float, float],
    legend_columns: int,
    legend_y: float,
    legend_fontsize: float,
    axes_top: float,
    stem: str,
    neutralize_deep_root: bool = False,
    branch_scale: float = 1.0,
    node_scale: float = 1.0,
    tip_scale: float = 1.0,
    bottom: float = .115,
    title_fontsize: float = 19,
    title_y: float = .985,
    subtitle1_y: float = .957,
    subtitle2_y: float = .938,
    show_note: bool = True,
    exact_canvas: bool = False,
    tick_rotation: float = 42,
    tick_ha: str = "right",
    subtitle_fontsize: float = 10.7,
    wrap_header: bool = False,
    left: float = .07,
    right: float = .975,
) -> None:
    tree, traits, styling, root_time, _, max_desc = prepared
    first_sample_date = pd.Timestamp(first_sample_date).normalize()
    last_date = pd.Timestamp(last_date).normalize()
    root_date = decimal_to_date(root_time)
    first_decimal = float(root_time)
    last_decimal = core.decimal_year(last_date)
    pad = max(20 / 365.25, (last_decimal - first_decimal) * .01)
    axis_left = first_decimal - pad
    axis_right = last_decimal + pad

    fig, ax = plt.subplots(figsize=figure_size, facecolor="white")
    draw_calendar_background(ax, first_decimal, last_decimal, band_years)
    ramps = {
        state: mcolors.LinearSegmentedColormap.from_list(
            f"calendar_ramp_{state}", ["#C5C9CC", color]
        )
        for state, color in colors.items()
    }

    def branch_width(descendants: int) -> float:
        return branch_scale * (.30 + 3.65 * np.sqrt(max(descendants, 1)) / np.sqrt(max_desc))

    def internal_node_size(descendants: int) -> float:
        return node_scale * (8 + 68 * np.sqrt(max(descendants, 1)) / np.sqrt(max_desc))

    for node in tree.Objects:
        x = root_time + float(node.height)
        parent = getattr(node, "parent", None)
        parent_x = root_time + float(parent.height) if parent is not None and parent is not node else x
        if x < first_decimal - 1e-5 or x > last_decimal + 1e-5:
            continue

        style = styling[node]
        state = str(style["state"])
        probability = float(style["probability"])
        ramp = ramps.get(state, next(iter(ramps.values())))
        neutral_root_stem = neutralize_deep_root and (
            node is tree.root
            or (parent is tree.root and str(style.get("source", "")) == "tip-majority")
        )
        color = "#989EA3" if neutral_root_stem else ramp(np.clip(probability, 0, 1))
        descendants = len(getattr(node, "leaves", []) or []) if node.branchType == "node" else 1
        width = branch_width(descendants)

        if node.branchType == "node" and node.children:
            ax.plot(
                [x, x], [node.children[-1].y, node.children[0].y],
                color=color, lw=width, zorder=4, solid_capstyle="round",
            )
            posterior = float(getattr(node, "traits", {}).get("posterior", 1.0))
            if posterior >= .5:
                ax.scatter(
                    x, node.y, s=internal_node_size(descendants),
                    facecolor="white", edgecolor=color, lw=.75, zorder=6,
                )
        else:
            tip_color = colors.get(state, "#9A9A9A")
            ax.scatter(x, node.y, s=tip_scale * 9.2, color="#2F3235", edgecolor="none", zorder=7, clip_on=False)
            ax.scatter(x, node.y, s=tip_scale * 6.2, color=tip_color, edgecolor="none", zorder=8, clip_on=False)

        branch_start = max(parent_x, first_decimal)
        if x >= branch_start:
            ax.plot(
                [branch_start, x], [node.y, node.y],
                color=color, lw=width, zorder=4, solid_capstyle="round",
            )

    ax.axvline(first_decimal, color="#4B5054", lw=.8, ls=(0, (3, 3)), alpha=.65, zorder=2)
    ax.axvline(last_decimal, color="#4B5054", lw=.8, ls=(0, (3, 3)), alpha=.65, zorder=2)
    ax.set_xlim(axis_left, axis_right)
    ax.set_ylim(-6, tree.ySpan + 8)
    ax.set_yticks([])
    ax.set_xlabel("Calendar date", labelpad=11)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#44484B")

    major_positions, major_labels, minor_positions = make_full_scale_ticks(
        first_decimal, last_date, tick_mode
    )
    ax.set_xticks(
        major_positions, major_labels, rotation=tick_rotation, ha=tick_ha,
        rotation_mode="anchor",
    )
    for label in ax.get_xticklabels():
        label.set_fontstyle("normal")
    ax.set_xticks(minor_positions, minor=True)
    ax.tick_params(axis="x", which="major", length=5.5, width=.9, pad=8, color="#3F4346")
    ax.tick_params(axis="x", which="minor", length=2.4, width=.65, color="#8A8E91")
    ax.grid(axis="x", which="major", color="#DDE0E2", lw=.55, alpha=.9, zorder=1)

    handles = [
        Line2D(
            [], [], color=colors[state], marker="o", lw=2.1,
            label=("Global non-USA" if state == "nonUSA" else state.replace("_", " ")),
        )
        for state in legend_order
    ]
    if neutralize_deep_root:
        handles.append(
            Line2D(
                [], [], color="#989EA3", marker="o", markerfacecolor="white", lw=2.4,
                label="Root / unsupported deepest stem (neutral)",
            )
        )
    handles.append(
        Line2D(
            [], [], color="#73787C", marker="o", markerfacecolor="white", lw=3,
            label="Node/branch size = descendant tips",
        )
    )
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(.055, legend_y),
        ncol=legend_columns, frameon=False, handlelength=2.0,
        columnspacing=1.15, labelspacing=.55,
        fontsize=legend_fontsize,
    )

    tip_count = len(traits)
    fig.suptitle(
        title, x=.055, ha="left", y=title_y, fontsize=title_fontsize,
        fontweight="bold", linespacing=.98,
    )
    subtitle1 = (
        f"{subtitle_label}\nFull root-to-tip scale: ~{root_date:%Y-%m-%d} to {last_date:%Y-%m-%d}"
        if wrap_header
        else f"{subtitle_label} | Full root-to-tip scale: ~{root_date:%Y-%m-%d} to {last_date:%Y-%m-%d}"
    )
    subtitle2 = (
        f"Sampling window: {first_sample_date:%Y-%m-%d} to {last_date:%Y-%m-%d}\nn = {tip_count:,} tips"
        if wrap_header
        else f"Sampling window: {first_sample_date:%Y-%m-%d} to {last_date:%Y-%m-%d} | n = {tip_count:,} tips"
    )
    fig.text(
        .055, subtitle1_y,
        subtitle1, fontsize=subtitle_fontsize, color="#5E6367", linespacing=1.12,
    )
    fig.text(
        .055, subtitle2_y,
        subtitle2, fontsize=subtitle_fontsize, color="#5E6367", linespacing=1.12,
    )
    note = (
        "Complete phylogenetic time scale from the HIPSTR root to the latest sampled tip. "
        "The root date is the point estimate implied by the summarized tree height."
    )
    if neutralize_deep_root:
        note += " The root and directly connected fallback stem are shown in neutral grey."
    if show_note:
        fig.text(.055, .018, note, fontsize=8.8, color="#5E6367")
    fig.subplots_adjust(left=left, right=right, top=axes_top, bottom=bottom)
    save_figure(fig, stem, tight=not exact_canvas)


def main() -> None:
    from two_state_phylogeny_landscape_11_69x5_neutral_root import main as draw_requested_tree
    draw_requested_tree()



if __name__ == "__main__":
    main()
