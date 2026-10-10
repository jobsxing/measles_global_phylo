
'Requested revised USA cap3 Panel A, using the original tree rendering function.'

import os

import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

from matplotlib.patches import FancyArrowPatch

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SCRIPT_DIR)

import visualize_USA_cap3_final_figures as base

import visualize_USA_cap3_posterior_fig1_subtrees as postfig

OUT = postfig.OUT

BIN_COLORS = postfig.BIN_COLORS

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei", "SimSun"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

def panel_a_tree_with_hist(subtrees, mj_counts):
    fig, ax = plt.subplots(figsize=(8.5 / 3.0 * 1.2, 13.2), facecolor="white")
    postfig.plot_full_tree_binary(
        ax,
        subtrees,
        show_node_markers=False,
        branch_width_base=0.4,
        branch_width_span=2.0,
        tip_halo_size=9.0,
        tip_size=7.6,
    )
    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()
    x0, x1 = ax.get_xlim()
    ax.set_xticks(range(int(np.ceil(x0 / 10) * 10), int(np.floor(x1 / 10) * 10) + 1, 10))
    ax.set_xlim(x0, x1)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title("A  HIPSTR tree recoded as\nUSA vs Global non-USA", fontsize=8, loc="left", fontweight="bold")

    handles = [
        mpl.lines.Line2D([], [], color=BIN_COLORS["Global non-USA"], marker="o", lw=1.6, markersize=4, label="Global non-USA"),
        mpl.lines.Line2D([], [], color=BIN_COLORS["USA"], marker="o", lw=1.6, markersize=4, label="USA"),
        mpl.lines.Line2D([], [], color="#777777", lw=2.4, label="Branch width =\ndescendant tips"),
    ]
    leg = ax.legend(
        handles=handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.995),
        fontsize=5.8,
        borderpad=0.35,
        labelspacing=0.5,
        handlelength=1.8,
    )
    leg.set_zorder(30)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.955, bottom=0.05)
    png = OUT / "USA_cap3_panel_A_tree_with_inset_hist_revised.png"
    pdf = OUT / "USA_cap3_panel_A_tree_with_inset_hist_revised.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf}")


def main():
    panel_a_tree_with_hist(None, None)


if __name__ == "__main__":
    main()
