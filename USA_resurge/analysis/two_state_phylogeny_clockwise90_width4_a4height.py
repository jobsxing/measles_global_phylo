from __future__ import annotations

import math

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import core_dta_visualization as core
from standalone_phylogeny_full_dates import (
    FIG,
    decimal_to_date,
    make_full_scale_ticks,
)


def draw_calendar_background_horizontal(
    ax: plt.Axes,
    first_decimal: float,
    last_decimal: float,
    band_years: int,
) -> None:
    first_band = int(math.floor(first_decimal / band_years) * band_years)
    for top_year in range(first_band, math.ceil(last_decimal) + band_years, band_years):
        if (top_year // band_years) % 2 == 0:
            top = max(first_decimal, float(top_year))
            bottom = min(last_decimal, float(top_year + band_years))
            if bottom > top:
                ax.axhspan(top, bottom, color="#EFEFEF", alpha=.78, zorder=0)


def draw_clockwise_rotated_tree(
    prepared,
    first_sample_date: pd.Timestamp,
    last_date: pd.Timestamp,
    *,
    figure_size: tuple[float, float] = (4.0, 11.69),
    stem: str = "2state_phylogeny_full_timescale_clockwise90_width4_a4height_neutral_root",
    title: str = "Two-state DTA phylogeny\nUSA vs global non-USA",
    title_x: float = .09,
    title_y: float = .985,
    title_fontsize: float = 13.5,
    view_label: str = "Clockwise 90-degree view",
    subtitle_x: float = .09,
    subtitle1_y: float = .915,
    subtitle2_y: float = .889,
    subtitle_fontsize: float = 7.7,
    legend_anchor: tuple[float, float] = (.085, .835),
    legend_columns: int = 1,
    legend_fontsize: float = 7.4,
    axes_left: float = .340,
    axes_right: float = .97,
    axes_top: float = .750,
    axes_bottom: float = .035,
    date_label_x: float = .035,
    date_label_y: float = .405,
    date_label_fontsize: float = 11.5,
) -> None:
    tree, traits, styling, root_time, _, max_desc = prepared
    colors = core.GLOBAL_COLORS
    first_sample_date = pd.Timestamp(first_sample_date).normalize()
    last_date = pd.Timestamp(last_date).normalize()
    root_date = decimal_to_date(root_time)
    first_decimal = float(root_time)
    last_decimal = core.decimal_year(last_date)
    pad = max(20 / 365.25, (last_decimal - first_decimal) * .01)

    fig, ax = plt.subplots(figsize=figure_size, facecolor="white")
    draw_calendar_background_horizontal(ax, first_decimal, last_decimal, 10)
    ramps = {
        state: mcolors.LinearSegmentedColormap.from_list(
            f"calendar_ramp_{state}", ["#C5C9CC", color]
        )
        for state, color in colors.items()
    }

    def branch_width(descendants: int) -> float:
        return .68 * (.30 + 3.65 * np.sqrt(max(descendants, 1)) / np.sqrt(max_desc))

    def internal_node_size(descendants: int) -> float:
        return .70 * (8 + 68 * np.sqrt(max(descendants, 1)) / np.sqrt(max_desc))

    for node in tree.Objects:
        y = root_time + float(node.height)
        parent = getattr(node, "parent", None)
        parent_y = root_time + float(parent.height) if parent is not None and parent is not node else y
        if y < first_decimal - 1e-5 or y > last_decimal + 1e-5:
            continue

        style = styling[node]
        state = str(style["state"])
        probability = float(style["probability"])
        ramp = ramps.get(state, next(iter(ramps.values())))
        neutral_root_stem = (
            node is tree.root
            or (parent is tree.root and str(style.get("source", "")) == "tip-majority")
        )
        color = "#989EA3" if neutral_root_stem else ramp(np.clip(probability, 0, 1))
        descendants = len(getattr(node, "leaves", []) or []) if node.branchType == "node" else 1
        width = branch_width(descendants)

        if node.branchType == "node" and node.children:
            ax.plot(
                [node.children[-1].y, node.children[0].y], [y, y],
                color=color, lw=width, zorder=4, solid_capstyle="round",
            )
            posterior = float(getattr(node, "traits", {}).get("posterior", 1.0))
            if posterior >= .5:
                ax.scatter(
                    node.y, y, s=internal_node_size(descendants),
                    facecolor="white", edgecolor=color, lw=.75, zorder=6,
                )
        else:
            tip_color = colors.get(state, "#9A9A9A")
            ax.scatter(node.y, y, s=.72 * 9.2, color="#2F3235", edgecolor="none", zorder=7, clip_on=False)
            ax.scatter(node.y, y, s=.72 * 6.2, color=tip_color, edgecolor="none", zorder=8, clip_on=False)

        branch_start = max(parent_y, first_decimal)
        if y >= branch_start:
            ax.plot(
                [node.y, node.y], [branch_start, y],
                color=color, lw=width, zorder=4, solid_capstyle="round",
            )

    ax.axhline(first_decimal, color="#4B5054", lw=.8, ls=(0, (3, 3)), alpha=.65, zorder=2)
    ax.axhline(last_decimal, color="#4B5054", lw=.8, ls=(0, (3, 3)), alpha=.65, zorder=2)
    ax.set_xlim(-6, tree.ySpan + 8)
    ax.set_ylim(first_decimal - pad, last_decimal + pad)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_ylabel("")
    ax.spines[["right", "top", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#44484B")

    major_positions, major_labels, minor_positions = make_full_scale_ticks(
        first_decimal, last_date, "decade"
    )
    ax.set_yticks(major_positions, major_labels)
    ax.set_yticks(minor_positions, minor=True)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontstyle("normal")
        label.set_ha("right")
    ax.tick_params(axis="y", which="major", length=5.2, width=.9, pad=5, labelsize=8.2, color="#3F4346")
    ax.tick_params(axis="y", which="minor", length=2.4, width=.65, color="#8A8E91")
    ax.grid(axis="y", which="major", color="#DDE0E2", lw=.55, alpha=.9, zorder=1)

    handles = [
        Line2D([], [], color=colors["nonUSA"], marker="o", lw=2.0, label="Global non-USA"),
        Line2D([], [], color=colors["USA"], marker="o", lw=2.0, label="USA"),
        Line2D(
            [], [], color="#989EA3", marker="o", markerfacecolor="white", lw=2.3,
            label="Neutral root / unsupported deepest stem",
        ),
        Line2D(
            [], [], color="#73787C", marker="o", markerfacecolor="white", lw=2.8,
            label="Node/branch size = descendant tips",
        ),
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=legend_anchor,
        ncol=legend_columns, frameon=False, handlelength=2.0,
        columnspacing=.9, labelspacing=.38, fontsize=legend_fontsize,
    )

    fig.suptitle(
        title,
        x=title_x, ha="left", y=title_y, fontsize=title_fontsize,
        fontweight="bold", linespacing=.98,
    )
    fig.text(
        subtitle_x, subtitle1_y,
        f"{view_label} | Full scale: ~{root_date:%Y-%m-%d} to {last_date:%Y-%m-%d}",
        fontsize=subtitle_fontsize, color="#5E6367",
    )
    fig.text(
        subtitle_x, subtitle2_y,
        f"Sampling: {first_sample_date:%Y-%m-%d} to {last_date:%Y-%m-%d} | n = {len(traits):,} tips",
        fontsize=subtitle_fontsize, color="#5E6367",
    )
    fig.text(
        date_label_x, date_label_y, "Calendar date", rotation=90,
        ha="center", va="center", fontsize=date_label_fontsize,
    )
    fig.subplots_adjust(
        left=axes_left, right=axes_right, top=axes_top, bottom=axes_bottom,
    )
    fig.savefig(FIG / f"{stem}.pdf", facecolor="white")
    fig.savefig(FIG / f"{stem}.png", dpi=340, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}", flush=True)


def main() -> None:
    prepared = core.prepare_tree(
        core.GLOBAL["hipstr"],
        core.GLOBAL["trees"],
        core.GLOBAL["traits"],
        "state2",
        "globalUSA_HIPSTR_posterior_state_recoding.csv",
    )
    traits = core.load_traits(core.GLOBAL["traits"])
    draw_clockwise_rotated_tree(prepared, traits.date.min(), traits.date.max())


if __name__ == "__main__":
    main()
