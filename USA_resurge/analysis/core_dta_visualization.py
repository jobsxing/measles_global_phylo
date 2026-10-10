from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from pathlib import Path

import baltic as bt
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


BASE = Path(__file__).resolve().parent / "WGS"
INTER = BASE
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures" / "final_core"
TAB = OUT / "tables" / "final_core"
for directory in (FIG, TAB):
    directory.mkdir(parents=True, exist_ok=True)

GLOBAL = {
    "hipstr": BASE / "geo_2state_globalUSA_combined.hipstr.tre",
    "trees": BASE / "geo_2state_globalUSA_combined.trees",
    "traits": BASE / "global_USA_2018plus_traits_2state.tsv",
}
REGIONAL = {
    "mj": BASE / "geo_census11_MJhistory.txt",
    "traits": BASE / "geo_census11_clade1367_traits.tsv",
    "bf": BASE / "Bayes.factor.test.result.csv",
}

GLOBAL_COLORS = {"nonUSA": "#E76F51", "USA": "#6A4C93"}
REGION_ORDER = [
    "Arizona", "East_North_Central", "East_other", "Mountain_other", "New_Mexico",
    "Pacific", "South_Carolina", "Texas", "Utah", "West_North_Central", "nonUSA",
]
REGION_COLORS = {
    "Arizona": "#D55E00",
    "East_North_Central": "#277DA1",
    "East_other": "#4D908E",
    "Mountain_other": "#9B5DE5",
    "New_Mexico": "#E9C46A",
    "Pacific": "#F3722C",
    "South_Carolina": "#43AA8B",
    "Texas": "#B23A48",
    "Utah": "#577590",
    "West_North_Central": "#90BE6D",
    "nonUSA": "#8D8D8D",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def save(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}", flush=True)


def hpd(values: np.ndarray | pd.Series, mass: float) -> tuple[float, float]:
    x = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    if not len(x):
        return np.nan, np.nan
    width = max(1, int(np.floor(mass * len(x))))
    if width >= len(x):
        return float(x[0]), float(x[-1])
    spans = x[width:] - x[: len(x) - width]
    start = int(np.argmin(spans))
    return float(x[start]), float(x[start + width])


def decimal_year(date: pd.Timestamp) -> float:
    return date.year + (date.dayofyear - 1) / (366 if date.is_leap_year else 365)


def load_traits(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    d.columns = ["taxon", "state"]
    extracted = d.taxon.astype(str).str.extract(r"_(\d{4})-(\d{2})(?:-(\d{2}))?$")
    extracted[2] = extracted[2].fillna("15")
    d["date"] = pd.to_datetime(extracted[0] + "-" + extracted[1] + "-" + extracted[2], errors="coerce")
    d["decimal_date"] = d.date.map(lambda x: decimal_year(x) if pd.notna(x) else np.nan)
    return d


def parse_translate(path: Path) -> dict[str, str]:
    translate: dict[str, str] = {}
    in_translate = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.lower() == "translate":
                in_translate = True
                continue
            if not in_translate:
                continue
            match = re.match(r"(\d+)\s+(.+?)[,;]$", stripped)
            if match:
                translate[match.group(1)] = match.group(2).strip().strip("'")
            if stripped.endswith(";"):
                break
    return translate


class CladeStateParser:
    def __init__(self, text: str, label_bits: dict[str, int], trait: str, targets: set[int]):
        self.text = re.sub(r"^\s*\[&R\]\s*", "", text.strip().rstrip(";"))
        self.i = 0
        self.label_bits = label_bits
        self.trait = trait
        self.targets = targets
        self.states: dict[int, str] = {}

    def peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def bracket(self) -> str:
        start = self.i
        depth = 0
        while self.i < len(self.text):
            char = self.text[self.i]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    self.i += 1
                    break
            self.i += 1
        return self.text[start:self.i]

    def suffix(self, signature: int) -> None:
        while self.peek() and self.peek() not in "[,():;":
            self.i += 1
        annotation = self.bracket() if self.peek() == "[" else ""
        match = re.search(rf'(?:^|[,\[&])\s*{re.escape(self.trait)}="([^"]+)"', annotation)
        if match and signature in self.targets:
            self.states[signature] = match.group(1)
        if self.peek() == ":":
            self.i += 1
            while self.peek() and self.peek() not in ",)":
                self.i += 1

    def subtree(self) -> int:
        if self.peek() == "(":
            self.i += 1
            signature = 0
            while True:
                signature |= self.subtree()
                if self.peek() == ",":
                    self.i += 1
                    continue
                if self.peek() == ")":
                    self.i += 1
                    break
            self.suffix(signature)
            return signature
        start = self.i
        while self.peek() and self.peek() not in "[,():;":
            self.i += 1
        label = self.text[start:self.i].strip().strip("'")
        signature = self.label_bits.get(label, 0)
        self.suffix(signature)
        return signature

    def parse(self) -> dict[int, str]:
        self.subtree()
        return self.states


def posterior_states_on_hipstr(
    hipstr_path: Path, trees_path: Path, traits_path: Path, trait_name: str, cache_name: str,
) -> tuple[object, pd.DataFrame, dict[object, dict[str, float | str]]]:
    cache = TAB / cache_name
    tree = bt.loadNexus(str(hipstr_path), absoluteTime=False)
    traits = load_traits(traits_path)
    trait_lookup = dict(zip(traits.taxon.astype(str), traits.state.astype(str)))
    tip_names = [str(n.name) for n in tree.Objects if n.branchType == "leaf"]
    bit_by_name = {name: 1 << i for i, name in enumerate(tip_names)}

    node_signature: dict[object, int] = {}
    node_tip_counts: dict[object, Counter] = {}

    def visit(node) -> tuple[int, Counter]:
        if node.branchType == "leaf":
            state = trait_lookup.get(str(node.name), "Unknown")
            signature, counts = bit_by_name[str(node.name)], Counter({state: 1})
        else:
            signature, counts = 0, Counter()
            for child in node.children:
                child_signature, child_counts = visit(child)
                signature |= child_signature
                counts.update(child_counts)
        node_signature[node] = signature
        node_tip_counts[node] = counts
        return signature, counts

    visit(tree.root)
    targets = set(node_signature.values())
    if cache.exists() and cache.stat().st_size > 20:
        state_summary = pd.read_csv(cache)
    else:
        translate = parse_translate(trees_path)
        label_bits = {label: bit_by_name[name] for label, name in translate.items() if name in bit_by_name}
        accum: dict[int, Counter] = defaultdict(Counter)
        trees_seen = 0
        with trees_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.lower().startswith("tree "):
                    continue
                newick = stripped.rsplit(" = ", 1)[1]
                states = CladeStateParser(newick, label_bits, trait_name, targets).parse()
                for signature, state in states.items():
                    accum[signature][state] += 1
                trees_seen += 1
        rows = []
        for signature in targets:
            counts = accum.get(signature, Counter())
            if counts:
                state, n = counts.most_common(1)[0]
                rows.append({"signature": str(signature), "state": state, "state_probability": n / sum(counts.values()),
                             "clade_match_probability": sum(counts.values()) / trees_seen, "matched_trees": sum(counts.values()),
                             "posterior_trees": trees_seen})
        state_summary = pd.DataFrame(rows)
        state_summary.to_csv(cache, index=False)
    by_signature = {int(r.signature): r for r in state_summary.itertuples(index=False)}
    styling = {}
    for node, signature in node_signature.items():
        record = by_signature.get(signature)
        if record is not None:
            styling[node] = {"state": record.state, "probability": float(record.state_probability),
                             "match_probability": float(record.clade_match_probability), "source": "posterior"}
        else:
            counts = node_tip_counts[node]
            state, n = counts.most_common(1)[0]
            styling[node] = {"state": state, "probability": n / sum(counts.values()),
                             "match_probability": 0.0, "source": "tip-majority"}
    return tree, traits, styling


def prepare_tree(
    hipstr_path: Path, trees_path: Path, traits_path: Path, trait_name: str, cache_name: str,
):
    tree, traits, styling = posterior_states_on_hipstr(hipstr_path, trees_path, traits_path, trait_name, cache_name)
    latest = float(traits.decimal_date.max())
    root_time = latest - float(tree.treeHeight)
    max_desc = max(len(getattr(n, "leaves", []) or []) for n in tree.Objects if n.branchType == "node")
    return tree, traits, styling, root_time, latest, max_desc


def draw_tree(
    ax: plt.Axes, prepared, colors: dict[str, str], xlim: tuple[float, float], title: str,
    legend_order: list[str], panel: str | None = None, compact_legend: bool = False,
) -> None:
    tree, traits, styling, root_time, latest, max_desc = prepared
    ramps = {state: mcolors.LinearSegmentedColormap.from_list(f"ramp_{state}", ["#C7C7C7", color])
             for state, color in colors.items()}

    def width(n: int) -> float:
        return .36 + 4.1 * np.sqrt(max(n, 1)) / np.sqrt(max_desc)

    def node_size(n: int) -> float:
        return 10 + 80 * np.sqrt(max(n, 1)) / np.sqrt(max_desc)

    for year in range(math.floor(xlim[0]), math.ceil(xlim[1])):
        if year % 2 == 0:
            ax.axvspan(year, year + 1, color="#EEF0F2", alpha=.55, zorder=0)
    for node in tree.Objects:
        x = root_time + float(node.height)
        parent = getattr(node, "parent", None)
        xp = root_time + float(parent.height) if parent is not None and parent is not node else x
        style = styling[node]
        state = str(style["state"])
        probability = float(style["probability"])
        color = ramps.get(state, ramps[next(iter(ramps))])(np.clip(probability, 0, 1))
        n_desc = len(getattr(node, "leaves", []) or []) if node.branchType == "node" else 1
        lw = width(n_desc)
        if node.branchType == "node" and node.children:
            ax.plot([x, x], [node.children[-1].y, node.children[0].y], color=color, lw=lw, zorder=4)
            posterior = float(getattr(node, "traits", {}).get("posterior", 1.0))
            if posterior >= .5 and x >= xlim[0]:
                ax.scatter(x, node.y, s=node_size(n_desc), facecolor="white", edgecolor=color, lw=.75, zorder=6)
        else:
            if x >= xlim[0]:
                ax.scatter(x, node.y, s=8.5, color="black", edgecolor="none", zorder=6)
                ax.scatter(x, node.y, s=5.8, color=colors.get(state, "#AAA"), edgecolor="none", zorder=7)
        ax.plot([max(xp, xlim[0]), x], [node.y, node.y], color=color, lw=lw, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(-6, tree.ySpan + 8)
    ax.set_yticks([])
    ax.set_xlabel("Calendar year")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_title(f"{panel + '  ' if panel else ''}{title}", loc="left", fontweight="bold", pad=8)
    handles = [Line2D([], [], color=colors[s], marker="o", lw=2.2,
                      label=("Global non-USA" if s == "nonUSA" else s.replace("_", " ")))
               for s in legend_order]
    handles.append(Line2D([], [], color="#777", marker="o", markerfacecolor="white", lw=3,
                          label="Node/branch size = descendant tips"))
    ncol = 2 if compact_legend else 1
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(.01, .995), ncol=ncol,
              frameon=True, facecolor="white", edgecolor="none", framealpha=.90,
              fontsize=8.2 if compact_legend else 9.3, handlelength=2.0, columnspacing=.8)


def full_route_counts(mj: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    trees = np.sort(mj.treeId.unique())
    locations = sorted(set(mj.startLocation) | set(mj.endLocation))
    routes = pd.MultiIndex.from_tuples([(a, b) for a in locations for b in locations if a != b],
                                      names=["startLocation", "endLocation"])
    counts = (mj[(mj.time >= start) & (mj.time <= end)]
              .groupby(["treeId", "startLocation", "endLocation"]).size().rename("count"))
    flat = counts.reindex(pd.MultiIndex.from_tuples(
        [(t, a, b) for t in trees for a, b in routes], names=["treeId", "startLocation", "endLocation"]),
        fill_value=0).reset_index()
    return flat


def temporal_direction_summary(mj_path: Path, start: float = 2011.0, end: float = 2026.999) -> tuple[pd.DataFrame, pd.DataFrame]:
    mj = pd.read_csv(mj_path)
    trees = np.sort(mj.treeId.unique())
    months = pd.date_range("2011-01-01", "2026-06-01", freq="MS")
    directions = [("nonUSA", "USA", "Global non-USA to USA"), ("USA", "nonUSA", "USA to global non-USA")]
    rows = []
    for a, b, label in directions:
        d = mj[(mj.startLocation == a) & (mj.endLocation == b) & (mj.time >= start) & (mj.time <= end)].copy()
        year = np.floor(d.time).astype(int)
        day = ((d.time - year) * 365.25).clip(0, 364)
        d["month"] = (pd.to_datetime(year.astype(str) + "-01-01") + pd.to_timedelta(day, unit="D")).dt.to_period("M").dt.to_timestamp()
        counts = d.groupby(["treeId", "month"]).size().rename("count").reindex(
            pd.MultiIndex.from_product([trees, months], names=["treeId", "month"]), fill_value=0).reset_index()
        counts["smoothed"] = counts.groupby("treeId")["count"].transform(
            lambda x: x.rolling(3, center=True, min_periods=1).mean()
        )
        counts["direction"] = label
        rows.append(counts)
    by_tree = pd.concat(rows, ignore_index=True)
    summary_rows = []
    for (direction, month), group in by_tree.groupby(["direction", "month"]):
        lo95, hi95 = hpd(group.smoothed, .95)
        lo50, hi50 = hpd(group.smoothed, .50)
        summary_rows.append({"direction": direction, "month": month, "mean": group.smoothed.mean(),
                             "median": group.smoothed.median(), "hpd50_low": lo50, "hpd50_high": hi50,
                             "hpd95_low": lo95, "hpd95_high": hi95})
    summary = pd.DataFrame(summary_rows)
    by_tree.to_csv(TAB / "globalUSA_temporal_MJ_by_tree_3month.csv", index=False)
    summary.to_csv(TAB / "globalUSA_temporal_MJ_mean_50_95_HPD.csv", index=False)
    return by_tree, summary


def draw_temporal_mj(ax: plt.Axes, summary: pd.DataFrame, title: str, panel: str | None = None) -> None:
    direction_colors = {"Global non-USA to USA": GLOBAL_COLORS["nonUSA"], "USA to global non-USA": GLOBAL_COLORS["USA"]}
    for direction, d in summary.groupby("direction"):
        d = d.sort_values("month")
        color = direction_colors[direction]
        ax.fill_between(d.month, d.hpd95_low, d.hpd95_high, color=color, alpha=.12, lw=0)
        ax.fill_between(d.month, d.hpd50_low, d.hpd50_high, color=color, alpha=.30, lw=0)
        ax.plot(d.month, d["mean"], color=color, lw=2.0, label=direction)
    ax.set_xlim(pd.Timestamp("2011-01-01"), pd.Timestamp("2026-06-30"))
    ax.set_ylabel("Monthly Markov jumps\n(3-month posterior smoothing)")
    ax.set_xlabel("Calendar year")
    ax.set_title(f"{panel + '  ' if panel else ''}{title}", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8", lw=.6)
    line_handles = [Line2D([], [], color=c, lw=2.4, label=k) for k, c in direction_colors.items()]
    interval_handles = [Line2D([], [], color="#555", lw=8, alpha=.30, label="50% HPD"),
                        Line2D([], [], color="#555", lw=8, alpha=.12, label="95% HPD")]
    ax.legend(handles=line_handles + interval_handles, frameon=False, ncol=2, loc="upper left")


def global_core(prepared, temporal: pd.DataFrame, counts: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10.2, 14.5), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[4.5, 1.45], hspace=.10)
    ax_tree = fig.add_subplot(grid[0])
    draw_tree(ax_tree, prepared, GLOBAL_COLORS, (2011.0, 2026.65),
              "HIPSTR phylogeny recoded as USA vs global non-USA", ["nonUSA", "USA"], panel="a")
    inset = inset_axes(ax_tree, width="36%", height="24%", loc="lower left", borderpad=1.8)
    imports = counts[(counts.startLocation == "nonUSA") & (counts.endLocation == "USA")]["count"]
    lo, hi = hpd(imports, .95)
    inset.hist(imports, bins=24, density=True, color="#575757", edgecolor="white", lw=.45)
    inset.axvline(imports.mean(), color=GLOBAL_COLORS["nonUSA"], lw=1.5)
    inset.axvline(lo, color="#222", ls=":", lw=1)
    inset.axvline(hi, color="#222", ls=":", lw=1)
    inset.set_title(f"Global non-USA to USA introductions\nmean {imports.mean():.1f}; 95% HPD {lo:.0f}-{hi:.0f}", fontsize=8)
    inset.set(xlabel="Jumps per posterior tree", ylabel="Density")
    inset.tick_params(labelsize=7)
    inset.spines[["top", "right"]].set_visible(False)
    ax_time = fig.add_subplot(grid[1])
    draw_temporal_mj(ax_time, temporal, "Bidirectional temporal Markov-jump dynamics, 2011-2026", panel="b")
    fig.subplots_adjust(left=.085, right=.985, top=.975, bottom=.055)
    save(fig, "final_core_2state_phylogeny_temporal_MJ", dpi=320)


def interregion_route_summary(mj_path: Path, bf_path: Path, start: float, end: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    mj = pd.read_csv(mj_path)
    full = full_route_counts(mj, start, end)
    years = end - start
    rows = []
    for (a, b), group in full.groupby(["startLocation", "endLocation"]):
        rate = group["count"] / years
        lo, hi = hpd(rate, .95)
        rows.append({"start": a, "end": b, "route": f"{a} to {b}", "mean_MJ_per_year": rate.mean(),
                     "median_MJ_per_year": rate.median(), "hpd95_low": lo, "hpd95_high": hi})
    route = pd.DataFrame(rows)
    bf = pd.read_csv(bf_path)
    route = route.merge(bf, left_on=["start", "end"], right_on=["start_name", "end_name"], how="left")
    route.sort_values(["bayes_factor", "mean_MJ_per_year"], ascending=False).to_csv(TAB / "interregion_MJ_per_year_BF.csv", index=False)
    full.to_csv(TAB / "interregion_MJ_counts_by_tree_since_first_tip.csv", index=False)
    return full, route


def draw_mj_bf(ax_mj: plt.Axes, ax_bf: plt.Axes, route: pd.DataFrame,
               mj_samples: pd.DataFrame | None = None, sample_years: float | None = None,
               top_n: int = 14,
               title: str = "Directional Markov jumps and posterior probability + BF support",
               panel: str | None = None,
               compact: bool = False) -> None:
    domestic = route[(route.start != "nonUSA") & (route.end != "nonUSA")].copy()
    supported = domestic[(domestic.bayes_factor >= 3) | (domestic.mean_MJ_per_year > 0)].copy()
    supported["rank_score"] = np.log10(supported.bayes_factor.clip(lower=1)) + np.log1p(supported.mean_MJ_per_year)
    top = supported.nlargest(min(top_n, len(supported)), "rank_score").sort_values("mean_MJ_per_year")
    y = np.arange(len(top))

    def tinted(color: str, white_fraction: float) -> str:
        rgb = np.asarray(mcolors.to_rgb(color))
        return mcolors.to_hex(rgb * (1 - white_fraction) + white_fraction)

    if mj_samples is not None and sample_years is not None:
        sample_groups = {
            key: group["count"].to_numpy(float) / sample_years
            for key, group in mj_samples.groupby(["startLocation", "endLocation"], sort=False)
        }
        for yi, row in zip(y, top.itertuples(index=False)):
            values = sample_groups[(row.start, row.end)]
            color = REGION_COLORS.get(row.start, "#6F7A82")
            light = tinted(color, .68)
            span = float(np.ptp(values))
            if span > 1e-10:
                pad = max(.06 * span, .03)
                grid = np.linspace(max(0, float(values.min()) - pad), float(values.max()) + pad, 260)
                density = gaussian_kde(values)(grid)
                density = density / density.max() * (.38 if compact else .48)
                ax_mj.fill_between(grid, yi, yi + density, color=light, alpha=.72, lw=0, zorder=1)
                ax_mj.plot(grid, yi + density, color=color, lw=.3, zorder=2)
            lo95, hi95 = hpd(values, .95)
            lo50, hi50 = hpd(values, .50)
            ax_mj.hlines(yi, lo95, hi95, color=light, lw=2.0 if compact else 2.4, zorder=3)
            ax_mj.hlines(yi, lo50, hi50, color=color, lw=4.6 if compact else 5.4, zorder=4)
            ax_mj.scatter(
                values.mean(), yi, s=28 if compact else 38,
                color=mcolors.to_hex(np.asarray(mcolors.to_rgb(color)) * .62),
                edgecolor="white", lw=.55, zorder=5,
            )
    else:
        source_colors = [REGION_COLORS.get(x, "#888") for x in top.start]
        ax_mj.hlines(y, top.hpd95_low, top.hpd95_high, color="#818181", lw=1.15)
        ax_mj.scatter(top.mean_MJ_per_year, y, c=source_colors, s=42,
                      edgecolor="#2D2D2D", lw=.45, zorder=3)
    ax_mj.set_yticks(y, [""] * len(top))
    from_x, to_x = -.22, -.015
    label_transform = ax_mj.get_yaxis_transform()
    route_fontsize = 8.4 if compact else 10.5
    header_fontsize = 9.5 if compact else 12.0
    for yi, start, end in zip(y, top.start, top.end):
        ax_mj.text(from_x, yi, start.replace("_", " "), transform=label_transform,
                   ha="right", va="center", fontsize=route_fontsize)
        ax_mj.text(to_x, yi, end.replace("_", " "), transform=label_transform,
                   ha="right", va="center", fontsize=route_fontsize)
    ax_mj.text(from_x, 1.012, "From", transform=ax_mj.transAxes,
               ha="right", va="bottom", fontsize=header_fontsize, fontweight="bold")
    ax_mj.text(to_x, 1.012, "To", transform=ax_mj.transAxes,
               ha="right", va="bottom", fontsize=header_fontsize, fontweight="bold")
    ax_mj.tick_params(axis="y", left=False)
    ax_mj.set_xlabel("Markov jumps/year" if compact else "Markov jumps per year")
    ax_mj.xaxis.label.set_size(10.5 if compact else 15.0)
    ax_mj.tick_params(axis="x", labelsize=9.5 if compact else 12.5)
    if title or panel:
        ax_mj.set_title(
            f"{panel + '  ' if panel else ''}{title or ''}", loc="left", fontweight="bold",
        )
    ax_mj.spines[["top", "right"]].set_visible(False)

    valid_prior = route[
        route.bayes_factor.gt(0)
        & route.posterior_probability.gt(0)
        & route.posterior_probability.lt(1)
    ]
    prior_odds = np.median(
        (valid_prior.posterior_probability / (1 - valid_prior.posterior_probability))
        / valid_prior.bayes_factor
    )
    probabilities = (.50, .75, .95)
    thresholds = [(probability / (1 - probability)) / prior_odds for probability in probabilities]
    support_colors = ["#D9D9D9", "#A6A6A6", "#666666", "#1F1F1F"]
    posterior = top.posterior_probability.to_numpy(float)
    support_class = np.digitize(posterior, probabilities)
    bar_colors = [support_colors[index] for index in support_class]

    bf_cap = 200.0
    bf_plot = np.minimum(top.bayes_factor.to_numpy(float), bf_cap)
    ax_bf.barh(y, bf_plot, color=bar_colors, height=.48, alpha=.96)
    for probability, threshold in zip(probabilities, thresholds):
        ax_bf.axvline(threshold, color="#686868", ls=(0, (4, 3)), lw=.9, zorder=3)
        ax_bf.text(
            threshold, 1.008, f"PP = {probability:.2f}",
            transform=ax_bf.get_xaxis_transform(), rotation=90,
            rotation_mode="anchor", ha="left", va="bottom",
            fontsize=7.0 if compact else 9.2, color="#4D4D4D", clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": .6, "alpha": .88},
        )
    ax_bf.set_yticks([])
    for yi, value in zip(y, top.bayes_factor):
        if value <= bf_cap:
            ax_bf.text(value + 3.0, yi, f"{value:.1f}", va="center", ha="left",
                       fontsize=7.4 if compact else 9.5, color="#3A3A3A")
    ax_bf.set_xlim(0, bf_cap)
    ax_bf.set_xticks([0, 50, 100, 150, 200])
    ax_bf.set_xlabel("Bayes factor (BF)")
    ax_bf.xaxis.label.set_size(10.5 if compact else 15.0)
    ax_bf.tick_params(axis="x", labelsize=9.5 if compact else 12.5)
    ax_bf.spines[["top", "right", "left"]].set_visible(False)
    ax_bf.tick_params(axis="y", left=False)

    halfeye_handles = [
        Line2D([], [], color="#B9C8D2", lw=2.4, label="95% HPD"),
        Line2D([], [], color="#527A96", lw=5.4, label="50% HPD"),
        Line2D([], [], marker="o", color="white", markerfacecolor="#263E4E",
               markersize=6.5, label="Posterior mean"),
    ]
    halfeye_legend = ax_mj.legend(
        handles=halfeye_handles, title="Half-eye summary", frameon=False,
        loc="center right", bbox_to_anchor=(.99, .42),
        fontsize=7.8 if compact else 10.0, title_fontsize=8.2 if compact else 10.8,
    )
    ax_mj.add_artist(halfeye_legend)
    bf_handles = [
        Patch(facecolor=support_colors[0], label=f"BF < {thresholds[0]:.1f}  (PP < .50)"),
        Patch(facecolor=support_colors[1], label=f"{thresholds[0]:.1f}-{thresholds[1]:.1f}  (PP .50-.75)"),
        Patch(facecolor=support_colors[2], label=f"{thresholds[1]:.1f}-{thresholds[2]:.1f}  (PP .75-.95)"),
        Patch(facecolor=support_colors[3], label=f"BF >= {thresholds[2]:.1f}  (PP >= .95)"),
    ]
    ax_mj.legend(
        handles=bf_handles, title="BF / posterior support", frameon=False,
        loc="lower right", bbox_to_anchor=(.99, .02),
        fontsize=7.2 if compact else 9.1, title_fontsize=7.8 if compact else 10.0,
    )


def markov_reward_summary(log_path: Path) -> pd.DataFrame:
    log = pd.read_csv(log_path, sep="\t", comment="#", low_memory=False)
    rows = []
    for col in [c for c in log if c.startswith("c_geo.") and c.endswith("_reward[1]")]:
        location = col.removeprefix("c_geo.").removesuffix("_reward[1]")
        lo50, hi50 = hpd(log[col], .50)
        lo95, hi95 = hpd(log[col], .95)
        rows.append({"location": location, "mean": log[col].mean(), "median": log[col].median(),
                     "hpd50_low": lo50, "hpd50_high": hi50, "hpd95_low": lo95, "hpd95_high": hi95})
    out = pd.DataFrame(rows).sort_values("mean")
    out.to_csv(TAB / "interregion_Markov_rewards_mean_50_95_HPD.csv", index=False)
    return out


def draw_rewards(ax: plt.Axes, rewards: pd.DataFrame, title: str = "Markov rewards by deme", panel: str | None = None) -> None:
    y = np.arange(len(rewards))
    colors = [REGION_COLORS.get(x, "#888") for x in rewards.location]
    ax.hlines(y, rewards.hpd95_low, rewards.hpd95_high, color="#A8A8A8", lw=2)
    ax.hlines(y, rewards.hpd50_low, rewards.hpd50_high, color=colors, lw=6, alpha=.75)
    ax.scatter(rewards["mean"], y, c=colors, s=55, edgecolor="white", lw=.65, zorder=4)
    ax.set_yticks(y, [x.replace("_", " ") for x in rewards.location])
    ax.set_xlabel("Markov reward (posterior mean, 50% and 95% HPD)")
    ax.set_title(f"{panel + '  ' if panel else ''}{title}", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#ECECEC", lw=.6)
    handles = [
        Line2D([], [], color="#A8A8A8", lw=2, label="95% HPD"),
        Line2D([], [], color="#606060", lw=6, alpha=.75, label="50% HPD"),
        Line2D([], [], marker="o", color="white", markerfacecolor="#606060", markersize=7, label="Posterior mean"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")


def regional_core(prepared, route: pd.DataFrame, rewards: pd.DataFrame,
                  mj_samples: pd.DataFrame, first_tip: float, last_tip: float) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 13.2), facecolor="white")
    draw_tree(ax, prepared, REGION_COLORS, (first_tip, last_tip + .12),
              "Interregional HIPSTR phylogeny with posterior ancestral-state recoding",
              REGION_ORDER, panel="a", compact_legend=True)
    ax.text(.995, .006, f"Display window starts at first sampled tip ({first_tip:.2f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555")
    fig.subplots_adjust(left=.08, right=.985, top=.97, bottom=.06)
    save(fig, "final_core_interregion_phylogeny", dpi=320)

    fig = plt.figure(figsize=(13.5, 6.8), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.1], wspace=.04)
    ax_mj, ax_bf = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    draw_mj_bf(
        ax_mj, ax_bf, route, mj_samples, last_tip + .999 / 365 - first_tip,
        top_n=18, title="", panel=None,
    )
    fig.subplots_adjust(left=.32, right=.98, top=.93, bottom=.12)
    save(fig, "final_core_interregion_MJ_BF")

    fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor="white")
    draw_rewards(ax, rewards, panel="c")
    fig.tight_layout()
    save(fig, "final_core_interregion_Markov_rewards")

    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=.18)
    ax_tree = fig.add_subplot(outer[0])
    draw_tree(ax_tree, prepared, REGION_COLORS, (first_tip, last_tip + .12),
              "Interregional phylogeny", REGION_ORDER, panel="a", compact_legend=True)
    right = outer[1].subgridspec(2, 2, height_ratios=[1.15, .9], width_ratios=[3.1, 1], hspace=.30, wspace=.05)
    ax_mj = fig.add_subplot(right[0, 0])
    ax_bf = fig.add_subplot(right[0, 1])
    draw_mj_bf(
        ax_mj, ax_bf, route, mj_samples, last_tip + .999 / 365 - first_tip,
        top_n=10, panel="b", compact=True,
    )
    ax_reward = fig.add_subplot(right[1, :])
    draw_rewards(ax_reward, rewards, panel="c")
    fig.suptitle("USA multi-deme discrete-trait analysis", fontsize=18, fontweight="bold", y=.995)
    fig.subplots_adjust(left=.06, right=.985, top=.95, bottom=.07)
    save(fig, "final_core_interregion_overview", dpi=320)


def main() -> None:
    regional_traits = load_traits(REGIONAL["traits"])
    first_tip = float(regional_traits.decimal_date.min())
    last_tip = float(regional_traits.decimal_date.max())
    mj_samples, route = interregion_route_summary(
        REGIONAL["mj"], REGIONAL["bf"], first_tip, last_tip + .999 / 365,
    )
    fig = plt.figure(figsize=(13.5, 6.8), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.1], wspace=.04)
    ax_mj, ax_bf = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    draw_mj_bf(
        ax_mj, ax_bf, route, mj_samples, last_tip + .999 / 365 - first_tip,
        top_n=18, title="", panel=None,
    )
    fig.subplots_adjust(left=.32, right=.98, top=.93, bottom=.12)
    save(fig, "final_core_interregion_MJ_BF")



if __name__ == "__main__":
    main()
