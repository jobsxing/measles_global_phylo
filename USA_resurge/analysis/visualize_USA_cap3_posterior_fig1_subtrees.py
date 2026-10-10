
'Binary HIPSTR tree renderer retained from the original cap3 code.'

import os

import re

import shutil

import sys

from pathlib import Path

import baltic as bt

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.colors as clr

import matplotlib.lines as mlines

import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

import numpy as np

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SCRIPT_DIR)

import visualize_USA_cap3_final_figures as base

OUT = Path(base.OUT)

BIN_COLORS = {"USA": "#6a4c93", "Global non-USA": "#e76f51"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei", "SimSun"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

def plot_full_tree_binary(
    ax,
    intro_major,
    show_node_markers=True,
    branch_width_base=0.45,
    branch_width_span=4.4,
    tip_halo_size=13.0,
    tip_size=9.5,
):
    tree = base.load_tree()
    ramps = {
        key: clr.LinearSegmentedColormap.from_list(f"binary_{key}", ["#B9B9B9", color], N=256)
        for key, color in BIN_COLORS.items()
    }

    leaf_counts = []
    for node in tree.Objects:
        if getattr(node, "branchType", None) == "node" or node.__class__.__name__.lower() == "node":
            leaf_counts.append(len(getattr(node, "leaves", []) or []))
    max_leaf_count = max(leaf_counts) if leaf_counts else 1

    def scaled_width(n_leaves):
        return branch_width_base + branch_width_span * np.sqrt(max(n_leaves, 1)) / np.sqrt(max_leaf_count)

    def scaled_node_size(n_leaves):
        return 18 + 95 * np.sqrt(max(n_leaves, 1)) / np.sqrt(max_leaf_count)

    for node in tree.Objects:
        x = getattr(node, "absoluteTime", 0.0)
        parent = getattr(node, "parent", None)
        xp = getattr(parent, "absoluteTime", x)
        y = node.y
        state = "USA" if base.node_state(node) == "USA" else "Global non-USA"
        color = ramps[state](np.clip(base.node_prob(node), 0, 1))
        n_leaves = len(getattr(node, "leaves", []) or []) if getattr(node, "branchType", None) == "node" else 1
        lw = scaled_width(n_leaves)
        is_leaf = (getattr(node, "branchType", None) == "leaf") or node.__class__.__name__.lower() == "leaf"
        is_internal = (getattr(node, "branchType", None) == "node") or node.__class__.__name__.lower() == "node"
        if is_internal:
            if getattr(node, "children", None):
                ax.plot([x, x], [node.children[-1].y, node.children[0].y], lw=lw, color=color, zorder=9)
            posterior = getattr(node, "traits", {}).get("posterior", 1.0)
            try:
                posterior = float(posterior)
            except Exception:
                posterior = 1.0
            if show_node_markers and posterior >= 0.5:
                ax.scatter(
                    x, y,
                    s=scaled_node_size(n_leaves),
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.85,
                    alpha=0.95,
                    zorder=20,
                )
        if is_leaf:
            if tip_halo_size > 0:
                ax.scatter(x, y, s=tip_halo_size, facecolor="black", edgecolor="none", zorder=10)
            ax.scatter(x, y, s=tip_size, facecolor=color, edgecolor="none", zorder=11)
        ax.plot([xp, x], [y, y], lw=lw, color=color, zorder=9)
    xs = [getattr(k, "absoluteTime", np.nan) for k in tree.Objects]
    xs = [v for v in xs if np.isfinite(v)]
    ax.set_xlim(max(1975, int(np.floor(min(xs) / 5) * 5)), max(2026.5, int(np.ceil(max(xs)))))
    base.draw_year_bands(ax)
    ax.set_xticks(range(int(ax.get_xlim()[0] // 5 * 5), int(np.ceil(ax.get_xlim()[1])) + 1, 5))
    ax.tick_params(axis="x", labelsize=9, size=3, width=0.7)
    ax.tick_params(axis="y", left=False, labelleft=False)
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    if hasattr(tree, "ySpan"):
        ax.set_ylim(-5, tree.ySpan + 18)
    ax.set_title("HIPSTR tree recoded as USA vs Global non-USA", fontsize=12, loc="left", fontweight="bold")
    handles = [
        mlines.Line2D([], [], color=BIN_COLORS["Global non-USA"], marker="o", lw=2, label="Global non-USA"),
        mlines.Line2D([], [], color=BIN_COLORS["USA"], marker="o", lw=2, label="USA"),
        mlines.Line2D([], [], color="#777777", marker="o", markerfacecolor="white", lw=3.5, markersize=8, label="Node/branch size = descendant tips"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=8)
