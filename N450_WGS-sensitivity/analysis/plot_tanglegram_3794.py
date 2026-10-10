"""Tanglegram for the main 3,794-taxon N450 tree: uncalibrated vs root-calibrated.

Mirrors Fig_tanglegram_cal_vs_uncal_N450_13deme, which does the same comparison
on the matched 901-taxon set. Palettes, branch widths, node sizes, decade
stripes and the greedy untangling all come from plot_tanglegrams_usa_style so
the two figures read the same way; only the tree pair, the taxon set and the
canvas height (3,794 tips instead of 901) differ.

There is no metadata TSV for the 3,794-taxon set, so the deme assignment and
the dates are taken from the BEAST XML itself - the same source the analysis
used - rather than re-derived from a sequence table.

Both summary trees are HIPSTR on the 13-deme BSSVS posterior:
  uncalibrated  revision_GLM/combined/bssvs13_uncal_HIPSTR.tre
  calibrated    revision_age.prior/main_3794/geo13cal3794_combined_HIPSTR.tre
"""
from __future__ import annotations

import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import baltic as bt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import plot_tanglegrams_usa_style as base

FS = 2.0

FIG_H = 34.0 * 2.0 / 3.0
TOP, BOTTOM = 0.893, 0.225
REF_AXES_H = 34.0 * (0.935 - 0.062)
SCALE = FIG_H * (TOP - BOTTOM) / REF_AXES_H

BASE = Path(__file__).resolve().parent
OUT = BASE

XML = BASE.parent / "xml" / "N450_3794_root_prior_change" / "geo13cal3794_rep_1.xml"
PATHS = {
    "uncal": BASE / "bssvs13_uncal_HIPSTR.tre",
    "cal": BASE / "geo13cal3794_combined_HIPSTR.tre",
}
ROOT_NOTE = {
    "uncal": "root 1975 [1968-1985]",
    "cal": "root 1948 [1945-1951]",
}
LABELS = {
    "uncal": "Main N450, 450 bp, n = 3,794\n(uncalibrated)",
    "cal": "Main N450, 450 bp, n = 3,794\n(root age calibrated to WGS)",
}
RESOLUTION = 13
QUESTION = ("What did calibrating the root actually change inside the main N450 tree?")


def build_metadata() -> pd.DataFrame:
    """taxon / deme13 / deme13_fullname / decimal_date / collection_date from the XML."""
    import analysis_helpers as old
    root = ET.parse(XML).getroot()
    label = old.STATE_LABEL
    rows = []
    for t in root.findall("./taxa/taxon"):
        attrs = {a.attrib["name"]: (a.text or "").strip() for a in t.findall("attr")}
        full = attrs.get("geography")
        code = old.VI_STATE_MAP.get(full)
        if code is None:
            raise ValueError(f"unmapped deme {full!r} on {t.attrib['id']}")
        d = float(t.find("date").attrib["value"])
        rows.append({"taxon": t.attrib["id"], "deme13": code,
                     "deme13_fullname": label[code], "decimal_date": d})
    md = pd.DataFrame(rows)
    if len(md) != 3794:
        raise ValueError(f"expected 3794 taxa, read {len(md)}")
    if set(md.deme13) - set(base.ORDER[RESOLUTION]):
        raise ValueError(f"unexpected demes {set(md.deme13) - set(base.ORDER[RESOLUTION])}")
    year = np.floor(md.decimal_date).astype(int)
    frac = md.decimal_date - year
    md["collection_date"] = (pd.to_datetime(year.astype(str) + "-01-01")
                             + pd.to_timedelta(frac * 365.2425, unit="D")).dt.normalize()
    return md.set_index("taxon", drop=False)


def load_tree(key: str, metadata: pd.DataFrame):
    t = bt.loadNexus(str(PATHS[key]), absoluteTime=False)
    tips = {str(n.name) for n in t.Objects if n.branchType == "leaf"}
    expected = set(metadata.index.astype(str))
    if tips != expected:
        raise ValueError(f"{key}: tip/metadata mismatch (tree={len(tips)}, "
                         f"metadata={len(expected)}, tree-only={len(tips - expected)}, "
                         f"metadata-only={len(expected - tips)})")
    return t


def draw_tree_scaled(ax, tree, y, metadata, title, invert_x):
    """base.draw_tree with every density-limited glyph size multiplied by SCALE.

    Line widths scale linearly, marker areas quadratically, so on the shorter
    canvas the drawing keeps the same glyph-to-tip-spacing ratio as the 34 in
    version. Everything else - palettes, ramps, decade stripes, tick locators -
    is taken from the shared module unchanged.
    """
    palette = base.PALETTES[RESOLUTION]
    states, probabilities, descendants = base.state_annotations(tree, metadata, RESOLUTION)
    max_desc = max(descendants.values())
    latest = float(metadata["decimal_date"].max())
    root_time = latest - float(tree.treeHeight)
    xmin, xmax = root_time, latest
    base.draw_background(ax, xmin, xmax)
    ramps = {s: mcolors.LinearSegmentedColormap.from_list(
        f"ramp_{RESOLUTION}_{s}", ["#BFC4C7", c]) for s, c in palette.items()}

    def width(n):
        return (0.20 + 2.35 * np.sqrt(max(n, 1)) / np.sqrt(max_desc)) * SCALE

    def node_size(n):
        return (4.0 + 31.0 * np.sqrt(max(n, 1)) / np.sqrt(max_desc)) * SCALE ** 2

    for node in tree.Objects:
        x = root_time + float(node.height)
        parent = getattr(node, "parent", None)
        parent_x = root_time + float(parent.height) if parent is not None and parent is not node else x
        color = ramps[states[node]](np.clip(probabilities[node], 0.0, 1.0))
        lw = width(descendants[node])
        if node.branchType == "node" and node.children:
            child_y = [y[child] for child in node.children]
            ax.plot([x, x], [min(child_y), max(child_y)], color=color, lw=lw,
                    solid_capstyle="round", zorder=3)
            if float(getattr(node, "traits", {}).get("posterior", 1.0)) >= 0.50:
                ax.scatter([x], [y[node]], s=node_size(descendants[node]),
                           facecolor="white", edgecolor=color, lw=0.55 * SCALE, zorder=5)
        else:
            ax.scatter([x], [y[node]], s=4.5 * SCALE ** 2, color="#2F3235",
                       edgecolor="none", zorder=6, clip_on=False)
            ax.scatter([x], [y[node]], s=2.7 * SCALE ** 2, color=palette[states[node]],
                       edgecolor="none", zorder=7, clip_on=False)
        ax.plot([parent_x, x], [y[node], y[node]], color=color, lw=lw,
                solid_capstyle="round", zorder=3)

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


def plot_pair(metadata, lt, rt, left_order, right_order, tau) -> None:
    left_y = base.node_layout(lt, left_order)
    right_y = base.node_layout(rt, right_order)
    ltip = {str(n.name): left_y[n] for n in lt.Objects if n.branchType == "leaf"}
    rtip = {str(n.name): right_y[n] for n in rt.Objects if n.branchType == "leaf"}

    fig = plt.figure(figsize=(16.0, FIG_H), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 0.56, 1.18],
                          left=0.055, right=0.955, top=TOP, bottom=BOTTOM, wspace=0.015)
    ax_left, ax_mid, ax_right = (fig.add_subplot(gs[0, i]) for i in range(3))

    lr = draw_tree_scaled(ax_left, lt, left_y, metadata, LABELS["uncal"], invert_x=False)
    rr = draw_tree_scaled(ax_right, rt, right_y, metadata, LABELS["cal"], invert_x=True)

    palette = base.PALETTES[RESOLUTION]
    field = f"deme{RESOLUTION}"
    for name in left_order:
        ax_mid.plot([0.0, 1.0], [ltip[name], rtip[name]],
                    color=palette[str(metadata.loc[name, field])],
                    lw=0.20 * SCALE, alpha=0.26, zorder=2)
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(-5, len(metadata) + 4)
    ax_mid.axis("off")

    first = metadata["collection_date"].min()
    last = metadata["collection_date"].max()
    fig.suptitle(f"Tanglegram of main-analysis summary phylogenies colored by "
                 f"{RESOLUTION} demes",
                 x=0.055, ha="left", y=0.989, fontsize=18, fontweight="bold")
    fig.text(0.055, 0.969, QUESTION, fontsize=13.0, color="#3A4044", style="italic")
    fig.text(0.055, 0.954,
             f"n = {len(metadata):,} identical isolates | connectors after greedy untangling | "
             f"Kendall tau of tip order = {tau:.2f}",
             fontsize=12.0, color="#4F5559")
    fig.text(0.055, 0.941,
             f"Sampling window: {first:%Y-%m-%d} to {last:%Y-%m-%d} | "
             f"left {ROOT_NOTE['uncal']}, axis ~{lr[0]:.0f}-{lr[1]:.0f} | "
             f"right {ROOT_NOTE['cal']}, axis ~{rr[0]:.0f}-{rr[1]:.0f}",
             fontsize=10.7, color="#676D71")

    handles = []
    for state in base.ORDER[RESOLUTION]:
        n = int((metadata[field] == state).sum())
        handles.append(Line2D([], [], color=palette[state], marker="o", lw=2.0,
                              label=f"{metadata.loc[metadata[field] == state, f'{field}_fullname'].iloc[0]} ({n})"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.078),
               ncol=3, frameon=False, handlelength=2.0, columnspacing=1.2)
    fig.text(0.055, 0.004, textwrap.fill(
        "White nodes mark posterior >= 0.50; node and branch size scale with the number of "
        "descendant tips. Branch color is the majority deme among descendant tips; color "
        "saturation is that majority proportion. Both panels share one time axis convention "
        "but are drawn to their own root; the calibrated tree is deeper by construction.", 108),
        fontsize=8.7 * FS, color="#676D71")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"Fig_tanglegram_cal_vs_uncal_Main3794_{RESOLUTION}deme"
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {stem.name}.{{pdf,png}}")


def main() -> None:
    sys.setrecursionlimit(60000)
    base.setup_plotting()
    plt.rcParams.update({'font.size': 11 * FS, 'axes.titlesize': 14 * FS,
                         'axes.labelsize': 13 * FS, 'xtick.labelsize': 11 * FS,
                         'ytick.labelsize': 11 * FS, 'legend.fontsize': 10.5 * FS})
    metadata = build_metadata()
    print(f"metadata: {len(metadata)} taxa, "
          f"{metadata.decimal_date.min():.3f} - {metadata.decimal_date.max():.3f}")
    lt = load_tree("uncal", metadata)
    rt = load_tree("cal", metadata)
    print(f"tree heights: uncalibrated {lt.treeHeight:.2f}, calibrated {rt.treeHeight:.2f}")
    lo, ro, tau = base.greedy_untangle(lt, rt)
    print(f"Kendall tau = {tau:.4f}")
    plot_pair(metadata, lt, rt, lo, ro, tau)


if __name__ == "__main__":
    main()
