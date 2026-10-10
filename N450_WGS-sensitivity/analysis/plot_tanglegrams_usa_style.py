from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import re

import baltic as bt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parent
DATA = ROOT
BEAST = ROOT
OUT = ROOT
METADATA = ROOT / "WGS_2011plus_noA_metadata_n450qc_sub150.tsv"
WGS_TREE = ROOT / "WGS_sub150_combined.hipstr.tre"
N450_TREE = ROOT / "N450_sub150_combined.hipstr.tre"

PALETTES = {
    6: {
        "AFR": "#E83E3E",
        "AMR": "#6A51A3",
        "EMR": "#F4A261",
        "EUR": "#2F6F91",
        "SEAR": "#2F7D58",
        "WPR": "#9E1B1F",
    },
    13: {
        "WEU": "#ED6A4A",
        "SEU": "#2A9D8F",
        "NEU": "#3B7EA1",
        "EEU": "#203A5F",
        "NAM": "#6A4C93",
        "SAM": "#80AA6B",
        "CHN": "#991B1E",
        "SOA": "#286A4B",
        "JAK": "#F4A261",
        "WEA": "#2A7F9E",
        "NAF": "#F18F01",
        "SAF": "#D62828",
        "OCE": "#8D99AE",
    },
}

ORDER = {
    6: ["AFR", "AMR", "EMR", "EUR", "SEAR", "WPR"],
    13: ["WEU", "SEU", "NEU", "EEU", "NAM", "SAM", "CHN", "SOA", "JAK", "WEA", "NAF", "SAF", "OCE"],
}


def setup_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10.5,
            "legend.fontsize": 9.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 400,
        }
    )


def load_inputs():
    metadata = pd.read_csv(METADATA, sep="\t", low_memory=False)
    metadata = metadata.set_index("taxon", drop=False)
    wgs = bt.loadNexus(str(WGS_TREE), absoluteTime=False)
    n450 = bt.loadNexus(str(N450_TREE), absoluteTime=False)
    left_names = {str(node.name) for node in wgs.Objects if node.branchType == "leaf"}
    right_names = {str(node.name) for node in n450.Objects if node.branchType == "leaf"}
    expected = set(metadata.index.astype(str))
    if left_names != right_names or left_names != expected:
        raise ValueError(
            "Tree/metadata tip mismatch: "
            f"WGS={len(left_names)}, N450={len(right_names)}, metadata={len(expected)}, "
            f"WGS-only={len(left_names - right_names)}, metadata-missing={len(left_names - expected)}"
        )
    return metadata, wgs, n450


def leaf_order(node) -> list[str]:
    if node.branchType == "leaf":
        return [str(node.name)]
    out: list[str] = []
    for child in node.children:
        out.extend(leaf_order(child))
    return out


def reorder_by_target(node, target_rank: dict[str, int]) -> list[str]:
    if node.branchType == "leaf":
        return [str(node.name)]
    descendants: dict[object, list[str]] = {}
    for child in node.children:
        descendants[child] = reorder_by_target(child, target_rank)
    node.children.sort(key=lambda child: float(np.median([target_rank[name] for name in descendants[child]])))
    return [name for child in node.children for name in descendants[child]]


def greedy_untangle(wgs, n450, iterations: int = 16) -> tuple[list[str], list[str], float]:
    left = leaf_order(wgs.root)
    right = leaf_order(n450.root)
    for _ in range(iterations):
        left_rank = {name: i for i, name in enumerate(left)}
        right = reorder_by_target(n450.root, left_rank)
        right_rank = {name: i for i, name in enumerate(right)}
        left = reorder_by_target(wgs.root, right_rank)
    left_rank = {name: i for i, name in enumerate(left)}
    right_rank = {name: i for i, name in enumerate(right)}
    tau = float(kendalltau([left_rank[name] for name in left], [right_rank[name] for name in left]).statistic)
    return left, right, tau


def node_layout(tree, order: list[str]) -> dict[object, float]:
    rank = {name: i for i, name in enumerate(order)}
    y: dict[object, float] = {}

    def visit(node) -> float:
        if node.branchType == "leaf":
            value = float(rank[str(node.name)])
        else:
            child_y = [visit(child) for child in node.children]
            value = float((child_y[0] + child_y[-1]) / 2.0)
        y[node] = value
        return value

    visit(tree.root)
    return y


def state_annotations(tree, metadata: pd.DataFrame, resolution: int):
    field = f"deme{resolution}"
    state: dict[object, str] = {}
    probability: dict[object, float] = {}
    descendants: dict[object, int] = {}

    def visit(node) -> Counter:
        if node.branchType == "leaf":
            counts = Counter({str(metadata.loc[str(node.name), field]): 1})
        else:
            counts = Counter()
            for child in node.children:
                counts.update(visit(child))
        majority, n = counts.most_common(1)[0]
        state[node] = majority
        descendants[node] = sum(counts.values())
        probability[node] = n / descendants[node]
        return counts

    visit(tree.root)
    return state, probability, descendants


def decimal_year_to_date_label(value: float) -> str:
    year = int(np.floor(value))
    return str(year)


def draw_background(ax: plt.Axes, xmin: float, xmax: float) -> None:
    first = int(np.floor(xmin / 10.0) * 10)
    for year in range(first, int(np.ceil(xmax)) + 1, 10):
        if (year // 10) % 2 == 0:
            ax.axvspan(year, year + 10, color="#EEF0F2", alpha=0.66, zorder=0)


def draw_tree(
    ax: plt.Axes,
    tree,
    y: dict[object, float],
    metadata: pd.DataFrame,
    resolution: int,
    title: str,
    invert_x: bool,
) -> tuple[float, float]:
    palette = PALETTES[resolution]
    states, probabilities, descendants = state_annotations(tree, metadata, resolution)
    max_desc = max(descendants.values())
    latest = float(metadata["decimal_date"].max())
    root_time = latest - float(tree.treeHeight)
    xmin, xmax = root_time, latest
    draw_background(ax, xmin, xmax)
    ramps = {
        state: mcolors.LinearSegmentedColormap.from_list(f"ramp_{resolution}_{state}", ["#BFC4C7", color])
        for state, color in palette.items()
    }

    def width(n: int) -> float:
        return 0.20 + 2.35 * np.sqrt(max(n, 1)) / np.sqrt(max_desc)

    def node_size(n: int) -> float:
        return 4.0 + 31.0 * np.sqrt(max(n, 1)) / np.sqrt(max_desc)

    for node in tree.Objects:
        x = root_time + float(node.height)
        parent = getattr(node, "parent", None)
        parent_x = root_time + float(parent.height) if parent is not None and parent is not node else x
        color = ramps[states[node]](np.clip(probabilities[node], 0.0, 1.0))
        lw = width(descendants[node])
        if node.branchType == "node" and node.children:
            child_y = [y[child] for child in node.children]
            ax.plot([x, x], [min(child_y), max(child_y)], color=color, lw=lw, solid_capstyle="round", zorder=3)
            posterior = float(getattr(node, "traits", {}).get("posterior", 1.0))
            if posterior >= 0.50:
                ax.scatter([x], [y[node]], s=node_size(descendants[node]), facecolor="white", edgecolor=color, lw=0.55, zorder=5)
        else:
            tip_color = palette[states[node]]
            ax.scatter([x], [y[node]], s=4.5, color="#2F3235", edgecolor="none", zorder=6, clip_on=False)
            ax.scatter([x], [y[node]], s=2.7, color=tip_color, edgecolor="none", zorder=7, clip_on=False)
        ax.plot([parent_x, x], [y[node], y[node]], color=color, lw=lw, solid_capstyle="round", zorder=3)

    ax.set_xlim(xmin, xmax)
    if invert_x:
        ax.invert_xaxis()
    ax.set_ylim(-5, len(metadata) + 4)
    ax.set_yticks([])
    ax.set_xlabel("Calendar year")
    ax.set_title(title, loc="center", fontweight="bold", pad=8)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#44484B")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="x", which="major", length=5.0, width=0.8, color="#44484B")
    ax.tick_params(axis="x", which="minor", length=2.2, width=0.55, color="#8A8E91")
    ax.grid(axis="x", which="major", color="#D9DCDE", lw=0.5, zorder=1)
    return xmin, xmax


def plot_tanglegram(metadata: pd.DataFrame, wgs, n450, resolution: int, left: list[str], right: list[str], tau: float) -> None:
    left_y = node_layout(wgs, left)
    right_y = node_layout(n450, right)
    left_tip_y = {str(node.name): left_y[node] for node in wgs.Objects if node.branchType == "leaf"}
    right_tip_y = {str(node.name): right_y[node] for node in n450.Objects if node.branchType == "leaf"}

    fig = plt.figure(figsize=(16.0, 15.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 0.56, 1.18], left=0.055, right=0.955, top=0.87, bottom=0.13, wspace=0.015)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    left_range = draw_tree(ax_left, wgs, left_y, metadata, resolution, "Whole genome (14,358 bp)", invert_x=False)
    right_range = draw_tree(ax_right, n450, right_y, metadata, resolution, "Extracted N450 (450 bp)", invert_x=True)

    palette = PALETTES[resolution]
    field = f"deme{resolution}"
    for name in left:
        state = str(metadata.loc[name, field])
        ax_mid.plot([0.0, 1.0], [left_tip_y[name], right_tip_y[name]], color=palette[state], lw=0.38, alpha=0.34, zorder=2)
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(-5, len(metadata) + 4)
    ax_mid.axis("off")

    first_date = pd.to_datetime(metadata["collection_date"], errors="coerce").min()
    last_date = pd.to_datetime(metadata["collection_date"], errors="coerce").max()
    fig.suptitle(
        f"Tanglegram of matched summary phylogenies colored by {resolution} "
        f"{'demes' if resolution == 13 else 'WHO regions'}",
        x=0.055,
        ha="left",
        y=0.982,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.946,
        f"n = {len(metadata):,} identical isolates | connectors after greedy untangling | Kendall tau of tip order = {tau:.2f}",
        fontsize=12.5,
        color="#4F5559",
    )
    fig.text(
        0.055,
        0.919,
        f"Sampling window: {first_date:%Y-%m-%d} to {last_date:%Y-%m-%d} | "
        f"tree scales: WGS ~{left_range[0]:.0f}-{left_range[1]:.0f}; N450 ~{right_range[0]:.0f}-{right_range[1]:.0f}",
        fontsize=10.7,
        color="#676D71",
    )

    handles = []
    for state in ORDER[resolution]:
        label_field = f"deme{resolution}_fullname"
        label = str(metadata.loc[metadata[field] == state, label_field].iloc[0]) if label_field in metadata.columns else state
        n = int((metadata[field] == state).sum())
        handles.append(Line2D([], [], color=palette[state], marker="o", lw=2.0, label=f"{label} ({n})"))
    handles.append(
        Line2D([], [], color="#7D8387", marker="o", markerfacecolor="white", lw=2.6, label="White nodes: posterior >= 0.50; node/branch size: descendant tips")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3 if resolution == 6 else 4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.text(
        0.055,
        0.014,
        "Branch color is the majority region among descendant tips; color saturation is that majority proportion. "
        "This is a visualization aid, not a re-estimated ancestral-state posterior.",
        fontsize=8.7,
        color="#676D71",
    )

    stem = OUT / f"Fig_tanglegram_{resolution}deme"
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_plotting()
    metadata, wgs, n450 = load_inputs()
    left, right, tau = greedy_untangle(wgs, n450)
    for resolution in [6, 13]:
        plot_tanglegram(metadata, wgs, n450, resolution, left, right, tau)


if __name__ == "__main__":
    main()
