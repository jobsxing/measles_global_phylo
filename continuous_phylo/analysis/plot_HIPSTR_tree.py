import re
from pathlib import Path

import baltic as bt
import matplotlib as mpl
import matplotlib.colors as clr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
TREE_PATH = BASE_DIR / "MeV_combine.discrete.HIPSTR.tree"
OUTPUT_PDF = BASE_DIR / "MeV_combine.discrete.HIPSTR.phylogeny.pdf"
OUTPUT_PNG = BASE_DIR / "MeV_combine.discrete.HIPSTR.phylogeny.preview.png"

TRAIT_NAME = "geography.rate"
ABS_TIME_STR = "2026-05-28"
TIP_REGEX = r"_(\d{4}-\d{2})(?:_|$)"

STATE_COLORS = {
    "WesternEurope": "#e76f51",
    "SouthEurope": "#2a9d8f",
    "NorthEurope": "#457b9d",
    "EasternEurope": "#1d3557",
    "NorthAmerica": "#6a4c93",
    "SouthAmerica": "#8ab17d",
    "China": "#8a171a",
    "JapanKorea": "#f4a261",
    "SouthAsia": "#2d6a4f",
    "WestAsia": "#277da1",
    "NorthAfrica": "#f77f00",
    "SubSaharanAfrica": "#d62828",
    "Oceania": "#8d99ae",
    "ancestor": "#B9B9B9",
}

STATE_ORDER = [
    "China",
    "EasternEurope",
    "JapanKorea",
    "NorthAfrica",
    "NorthAmerica",
    "NorthEurope",
    "Oceania",
    "SouthAmerica",
    "SouthAsia",
    "SouthEurope",
    "SubSaharanAfrica",
    "WestAsia",
    "WesternEurope",
]

STATE_LABELS = {
    "China": "China",
    "EasternEurope": "Eastern Europe",
    "JapanKorea": "Japan/Korea",
    "NorthAfrica": "North Africa",
    "NorthAmerica": "North America",
    "NorthEurope": "North Europe",
    "Oceania": "Oceania",
    "SouthAmerica": "South America",
    "SouthAsia": "South Asia",
    "SouthEurope": "South Europe",
    "SubSaharanAfrica": "Sub-Saharan Africa",
    "WestAsia": "West Asia",
    "WesternEurope": "Western Europe",
}


mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei", "SimSun"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


def canon_state(state):
    if state is None:
        return None
    return re.sub(r"[\s_/\-]+", "", str(state))


def decimal_date(date_string):
    return bt.decimalDate(date_string)


def load_tree():
    tree = bt.loadNexus(
        str(TREE_PATH),
        tip_regex=TIP_REGEX,
        date_fmt="%Y-%m",
        variableDate=False,
    )
    tree.setAbsoluteTime(decimal_date(ABS_TIME_STR))
    return tree


def trait_map_value(node, trait):
    value = getattr(node, "traits", {}).get(trait)
    return canon_state(value) if value is not None else None


def trait_prob_value(node, trait):
    value = getattr(node, "traits", {}).get(f"{trait}.prob")
    try:
        return float(value) if value is not None else 1.0
    except Exception:
        return 1.0


def trait_state_probs_from_node(node, trait):
    traits = getattr(node, "traits", {})

    states = traits.get(f"{trait}.set")
    probs = traits.get(f"{trait}.set.prob")
    if isinstance(states, (list, tuple)) and isinstance(probs, (list, tuple)) and len(states) == len(probs):
        out = {canon_state(s): float(p) for s, p in zip(states, probs)}
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
        if out:
            map_state, map_prob = max(out.items(), key=lambda item: item[1])
            return out, map_state, map_prob

    map_state = canon_state(traits.get(trait))
    map_prob = trait_prob_value(node, trait)
    return ({map_state: map_prob} if map_state else {}), map_state, map_prob


def scan_states(tree, trait):
    states = set()
    for node in tree.Objects:
        map_state = trait_map_value(node, trait)
        if map_state:
            states.add(map_state)
        trait_set = getattr(node, "traits", {}).get(f"{trait}.set")
        if isinstance(trait_set, (list, tuple)):
            states.update(canon_state(s) for s in trait_set)
    return sorted(s for s in states if s)


def build_color_map(states):
    color_map = dict(STATE_COLORS)
    auto_colors = plt.get_cmap("tab20").colors
    missing = [s for s in states if s not in color_map and s != "ancestor"]
    for idx, state in enumerate(missing):
        color_map[state] = clr.to_hex(auto_colors[idx % len(auto_colors)])
    return color_map


def build_order(states):
    ordered = [state for state in STATE_ORDER if state in states]
    ordered.extend(state for state in sorted(states) if state not in ordered)
    return ordered


def state_ramp(state, colors_map, cache):
    if state not in cache:
        cache[state] = clr.LinearSegmentedColormap.from_list(
            f"ramp_{state}", ["#B9B9B9", colors_map.get(state, "#999999")], N=256
        )
    return cache[state]


def draw_year_bands(ax, x_min, x_max, step_years=5, alpha=0.065):
    start = int(np.floor(x_min / step_years) * step_years)
    idx = 0
    for year in range(start, int(np.ceil(x_max)) + step_years, step_years):
        if idx % 2 == 0:
            ax.axvspan(year, min(year + step_years, x_max), color="k", alpha=alpha, zorder=0)
        idx += 1


def draw_tree(tree, colors_map, order):
    fig, ax = plt.subplots(figsize=(7.2, 7.47), facecolor="white", constrained_layout=False)
    root = next((node for node in tree.Objects if getattr(node, "parent", None) is None), tree.Objects[0])

    node_probs = {}
    node_states = {}
    for node in tree.Objects:
        node_states[node] = trait_map_value(node, TRAIT_NAME)
        node_probs[node] = trait_prob_value(node, TRAIT_NAME)

    ramp_cache = {}
    branch_width = 0.72
    tip_size = 7.0
    node_size = 15.0

    x_values = [getattr(node, "absoluteTime", None) for node in tree.Objects]
    x_values = [x for x in x_values if x is not None]
    x_min = np.floor(min(x_values) / 5) * 5
    x_max = np.ceil(max(x_values)) + 0.8

    draw_year_bands(ax, x_min, x_max)

    for node in tree.Objects:
        x = getattr(node, "absoluteTime", 0.0)
        parent = getattr(node, "parent", None)
        xp = getattr(parent, "absoluteTime", x)
        y = node.y

        state = node_states.get(node)
        prob = float(np.clip(node_probs.get(node, 1.0), 0.0, 1.0))
        color = state_ramp(state, colors_map, ramp_cache)(prob)
        width = branch_width

        is_leaf = getattr(node, "branchType", None) == "leaf" or node.__class__.__name__.lower() == "leaf"
        is_internal = getattr(node, "branchType", None) == "node" or node.__class__.__name__.lower() == "node"

        if is_internal:
            width = branch_width + len(getattr(node, "leaves", [])) * 0.00045
            children = getattr(node, "children", None)
            if children:
                ax.plot([x, x], [children[-1].y, children[0].y], lw=width, color=color, zorder=9)
            posterior = getattr(node, "traits", {}).get("posterior", 0.0)
            if posterior >= 0.5:
                ax.scatter(x, y, s=node_size, facecolor="white", edgecolor=color, linewidth=0.55, zorder=12)

        ax.plot([xp, x], [y, y], lw=width, color=color, zorder=9)

        if is_leaf:
            ax.scatter(x, y, s=tip_size * 1.45, facecolor="#2b2b2b", edgecolor="none", zorder=10)
            ax.scatter(x, y, s=tip_size, facecolor=color, edgecolor="none", zorder=11)

    ax.set_xlim(x_min, x_max)
    if hasattr(tree, "ySpan"):
        ax.set_ylim(-7, tree.ySpan + 7)

    tick_start = int(np.floor(x_min / 5) * 5)
    tick_end = int(np.ceil(x_max / 5) * 5)
    years = list(range(tick_start, tick_end + 1, 5))
    ax.set_xticks(years)
    ax.set_xticklabels([str(year) for year in years], fontsize=9)
    ax.tick_params(axis="x", size=4.5, width=0.3, color="#000000", pad=2)
    ax.tick_params(axis="y", labelsize=0, size=0)
    ax.set_yticklabels([])

    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.3)

    handles = []
    for state in order:
        color = colors_map.get(state)
        if not color:
            continue
        label = STATE_LABELS.get(state, state)
        handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                markerfacecolor=color,
                markeredgecolor=color,
                lw=1.0,
                label=label,
                markersize=5.5,
            )
        )
    if handles:
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.018, 0.038),
            frameon=False,
            markerfirst=True,
            prop={"size": 7.2},
            handlelength=1.8,
            labelspacing=0.42,
            borderaxespad=0.0,
        )

    plt.subplots_adjust(left=0.028, right=0.985, bottom=0.055, top=0.992)
    return fig


def main():
    tree = load_tree()
    states = scan_states(tree, TRAIT_NAME)
    colors_map = build_color_map(states)
    order = build_order(states)
    fig = draw_tree(tree, colors_map, order)
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"PDF: {OUTPUT_PDF}")
    print(f"Preview: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
