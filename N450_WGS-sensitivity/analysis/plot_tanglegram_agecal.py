"""Tanglegrams for the root-age-calibrated matched N450 tree.

Two comparisons, both on the identical 901-taxon set:
  A  WGS (uncalibrated)          vs  N450 (root-calibrated)
  B  N450 (uncalibrated)         vs  N450 (root-calibrated)

Everything about the rendering - palettes, branch widths, node sizes, decade
stripes, untangling - is reused from plot_tanglegrams_usa_style so these figures
are directly comparable with the published ones. Only the tree pair, the panel
titles and the output name are parameterised here.
"""
from __future__ import annotations

from pathlib import Path

import baltic as bt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import plot_tanglegrams_usa_style as base

METADATA = base.METADATA
OUT = Path(__file__).resolve().parent

WGS_UNCAL = base.BEAST / "WGS_sub150_combined.hipstr.tre"
N450_UNCAL = base.BEAST / "N450_sub150_combined.hipstr.tre"
N450_CAL = base.BEAST / "geo13cal_combined.trees.HIPSTR.tree"

ROOT_NOTE = {
    "wgs_uncal": "root 1947 [1930-1964]",
    "n450_uncal": "root 1973 [1965-1982]",
    "n450_cal": "root 1948 [1945-1951]",
}
LABELS = {
    "wgs_uncal": "Whole genome, 14,358 bp\n(uncalibrated)",
    "n450_uncal": "Extracted N450, 450 bp\n(uncalibrated)",
    "n450_cal": "Extracted N450, 450 bp\n(root age calibrated to WGS)",
}
PATHS = {"wgs_uncal": WGS_UNCAL, "n450_uncal": N450_UNCAL, "n450_cal": N450_CAL}

COMPARISONS = [
    ("cal_vs_WGS", "wgs_uncal", "n450_cal",
     "Does calibrating the N450 root reproduce the whole-genome phylogeny?"),
    ("cal_vs_uncal", "n450_uncal", "n450_cal",
     "What did calibrating the root actually change inside the N450 tree?"),
]


def load_metadata() -> pd.DataFrame:
    md = pd.read_csv(METADATA, sep="\t", low_memory=False)
    return md.set_index("taxon", drop=False)


def load_tree(key: str, metadata: pd.DataFrame):
    t = bt.loadNexus(str(PATHS[key]), absoluteTime=False)
    tips = {str(n.name) for n in t.Objects if n.branchType == "leaf"}
    expected = set(metadata.index.astype(str))
    if tips != expected:
        raise ValueError(f"{key}: tip/metadata mismatch "
                         f"(tree={len(tips)}, metadata={len(expected)}, "
                         f"tree-only={len(tips - expected)})")
    return t


def plot_pair(metadata, left_tree, right_tree, left_key, right_key,
              resolution, left_order, right_order, tau, tag, question) -> None:
    left_y = base.node_layout(left_tree, left_order)
    right_y = base.node_layout(right_tree, right_order)
    left_tip_y = {str(n.name): left_y[n] for n in left_tree.Objects if n.branchType == "leaf"}
    right_tip_y = {str(n.name): right_y[n] for n in right_tree.Objects if n.branchType == "leaf"}

    fig = plt.figure(figsize=(16.0, 15.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 0.56, 1.18],
                          left=0.055, right=0.955, top=0.845, bottom=0.13, wspace=0.015)
    ax_left, ax_mid, ax_right = (fig.add_subplot(gs[0, i]) for i in range(3))

    lr = base.draw_tree(ax_left, left_tree, left_y, metadata, resolution,
                        LABELS[left_key], invert_x=False)
    rr = base.draw_tree(ax_right, right_tree, right_y, metadata, resolution,
                        LABELS[right_key], invert_x=True)

    palette = base.PALETTES[resolution]
    field = f"deme{resolution}"
    for name in left_order:
        ax_mid.plot([0.0, 1.0], [left_tip_y[name], right_tip_y[name]],
                    color=palette[str(metadata.loc[name, field])],
                    lw=0.38, alpha=0.34, zorder=2)
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(-5, len(metadata) + 4)
    ax_mid.axis("off")

    first = pd.to_datetime(metadata["collection_date"], errors="coerce").min()
    last = pd.to_datetime(metadata["collection_date"], errors="coerce").max()
    fig.suptitle(f"Tanglegram of matched summary phylogenies colored by {resolution} "
                 f"{'demes' if resolution == 13 else 'WHO regions'}",
                 x=0.055, ha="left", y=0.982, fontsize=18, fontweight="bold")
    fig.text(0.055, 0.955, question, fontsize=13.0, color="#3A4044", style="italic")
    fig.text(0.055, 0.934,
             f"n = {len(metadata):,} identical isolates | connectors after greedy untangling | "
             f"Kendall tau of tip order = {tau:.2f}",
             fontsize=12.0, color="#4F5559")
    fig.text(0.055, 0.913,
             f"Sampling window: {first:%Y-%m-%d} to {last:%Y-%m-%d} | "
             f"left {ROOT_NOTE[left_key]}, axis ~{lr[0]:.0f}-{lr[1]:.0f} | "
             f"right {ROOT_NOTE[right_key]}, axis ~{rr[0]:.0f}-{rr[1]:.0f}",
             fontsize=10.7, color="#676D71")

    handles = []
    for state in base.ORDER[resolution]:
        lf = f"deme{resolution}_fullname"
        label = (str(metadata.loc[metadata[field] == state, lf].iloc[0])
                 if lf in metadata.columns else state)
        handles.append(Line2D([], [], color=palette[state], marker="o", lw=2.0,
                              label=f"{label} ({int((metadata[field] == state).sum())})"))
    handles.append(Line2D([], [], color="#7D8387", marker="o", markerfacecolor="white", lw=2.6,
                          label="White nodes: posterior >= 0.50; node/branch size: descendant tips"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=3 if resolution == 6 else 4, frameon=False,
               handlelength=2.0, columnspacing=1.2)
    fig.text(0.055, 0.014,
             "Branch color is the majority region among descendant tips; color saturation is that "
             "majority proportion. Both panels share one time axis convention but are drawn to their "
             "own root; the calibrated tree is deeper by construction.",
             fontsize=8.7, color="#676D71")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"Fig_tanglegram_{tag}_{resolution}deme"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {stem.name}.{{pdf,png,svg}}")


def main() -> None:
    base.setup_plotting()
    metadata = load_metadata()
    for tag, left_key, right_key, question in COMPARISONS:
        print(f"[{tag}] {left_key} (left) vs {right_key} (right)")
        for resolution in (13, 6):
            lt = load_tree(left_key, metadata)
            rt = load_tree(right_key, metadata)
            lo, ro, tau = base.greedy_untangle(lt, rt)
            print(f"  {resolution}-deme: Kendall tau = {tau:.3f}")
            plot_pair(metadata, lt, rt, left_key, right_key, resolution, lo, ro,
                      tau, tag, question)


if __name__ == "__main__":
    main()
