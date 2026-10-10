"""Shared code extracted from saptiotemporal_new.ipynb, cells 2, 4 and 5 (one-based)."""
import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei", "SimSun"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


import re
import numpy as np
import pandas as pd
from pathlib import Path

BF_MIN_C1 = 100
SPANNING_MAX_YEARS_C1 = 2

def is_leap_year(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

def decimal_date(ts) -> float:
    ts = pd.to_datetime(ts)
    year = ts.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end   = pd.Timestamp(year=year + 1, month=1, day=1)
    total = (end - start).days
    return year + (ts - start).days / total

def date_decimal(dec: float) -> pd.Timestamp:
    year = int(np.floor(dec))
    frac = float(dec) - year
    days = 366 if is_leap_year(year) else 365
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=frac * days)

def parse_location(loc: str):
    """
    兼容几种写法：
    - '2015_Endemic'
    - '2015_Region_eliminated'
    - '2015-China_endemic'
    - '2015-China'
    解析失败时返回 (None, None)
    """
    s = str(loc)
    parts = s.split("_")
    if len(parts) == 2 and parts[1].lower() == "endemic":
        return int(parts[0]), "Endemic"
    if len(parts) >= 3 and parts[-1].lower() == "eliminated":
        return int(parts[0]), parts[1]
    m = re.match(r"^(\d{4})[_\-](.*)$", s)
    if m:
        y = int(m.group(1)); tail = m.group(2)
        return y, ("Endemic" if "endemic" in tail.lower() else tail)
    return None, None

def status_of(loc: str) -> str:
    """
    对 location 粗略归类为 endemic / eliminated / unknown
    （现在主要用于 evidence_code，可保留）
    """
    s = str(loc).lower().strip()
    if s.endswith("endemic") or "endemic" in s:
        return "endemic"
    if s.endswith("eliminated") or "eliminated" in s:
        return "eliminated"
    return "unknown"

def evidence_code(src_loc: str, dst_loc: str) -> str:
    """
    仅用于着色 legend：I–IV / NA
    """
    src = status_of(src_loc)
    dst = status_of(dst_loc)
    if src=="endemic" and dst=="endemic":     return "I"
    if src=="endemic" and dst=="eliminated":  return "II"
    if src=="eliminated" and dst=="endemic":  return "III"
    if src=="eliminated" and dst=="eliminated": return "IV"
    return "NA"


import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def plot_cluster_bands(
    pres_df, edges_df, years, rows,
    *,
    CELL_W=8, CELL_H=2, YEAR_GAP=2.0, BUBBLE_PT=12,
    MAX_YEAR_SPAN=6, BOX_ASPECT=0.12,
    REGION_COLORS=None,
    EVIDENCE_COLOR=None,
    BF_EDGE_COLOR=None,
    YEAR_BAND_COLORS=("#edf2f4", "#8d99ae"),
    LINEWIDTH_MIN=0.5, LINEWIDTH_MAX=5,
    USE_QUANTILES=False, MANUAL_THRESHOLDS=(1,3,6),
    CURVE_BASE=0.25, CURVE_GAIN=0.10,
    ARROW_HEAD=14, ARROW_ALPHA=1.0,
    OUTLINE_MIN=2.0, OUTLINE_MARGIN=1.0,
    title="Genotype B — presence bubbles + jump edges (BF ≥ 100)",
    save_path=None, dpi=600
):
    """
    pres_df: 必含 ['year','region']
    edges_df: 必含 ['startYear','endYear','startRegion','endRegion','mean_jumps','bayes_factor','year_span','evidence']
    years: list[int]
    rows:  建议顺序为 ['Americas','Europe','Asia','Oceania','Endemic']
           （只影响气泡绘制顺序，不影响坐标）
    """

    need_pres = {"year","region"}
    miss_pres = need_pres - set(pres_df.columns)
    if miss_pres:
        raise ValueError(f"pres_df 缺少字段: {sorted(miss_pres)}")

    need_edges = {"startYear","endYear","startRegion","endRegion","mean_jumps","bayes_factor","year_span","evidence"}
    miss_edges = need_edges - set(edges_df.columns)
    if miss_edges:
        raise ValueError(f"edges_df 缺少字段: {sorted(miss_edges)}")

    if REGION_COLORS is None:
        REGION_COLORS = {
            "Europe":   "#2a9d8f",
            "Americas": "#936639",
            "Endemic":  "#fdf0d5",
            "Asia":     "#c1121f",
            "Oceania":  "#780000",
        }
    if EVIDENCE_COLOR is None:
        EVIDENCE_COLOR = {"I":"#3c6e71","II":"#284b63","III":"#c9184a","IV":"#800f2f","NA":"#353535"}
    if BF_EDGE_COLOR is None:
        BF_EDGE_COLOR  = {"100-200":"#F2F2F2", ">200":"#000000"}

    def col(r): return REGION_COLORS.get(r, "#888888")

    subcols = np.array([0.5, 1.5, 2.5, 3.5]) * CELL_W
    row_y = {
        "continents": 1.6 * CELL_H,
        "endemic":    0.4 * CELL_H,
    }

    slot = {
        "Americas": ("continents", 0),
        "Europe":   ("continents", 1),
        "Asia":     ("continents", 2),
        "Oceania":  ("continents", 3),
        "Endemic":  ("endemic", "center"),
    }

    YEAR_WIDTH = 4 * CELL_W + YEAR_GAP
    xticks_pos = [j * YEAR_WIDTH + 2.0 * CELL_W for j in range(len(years))]

    def make_lw_mapper(df_edges):
        vals = df_edges["mean_jumps"].values
        if len(vals) == 0:
            return lambda v: LINEWIDTH_MIN
        if USE_QUANTILES and len(vals) >= 4:
            q1, q2, q3 = np.quantile(vals, [0.25, 0.50, 0.75]); thr = [q1, q2, q3]
        else:
            thr = MANUAL_THRESHOLDS
        def lw(v):
            if v <= thr[0]: t = 0.10
            elif v <= thr[1]: t = 0.35
            elif v <= thr[2]: t = 0.75
            else: t = 1.00
            return LINEWIDTH_MIN + (LINEWIDTH_MAX - LINEWIDTH_MIN) * t
        return lw

    def bf_edge_color(bf):
        if bf is None: return None
        try:
            bfv = float(bf)
        except Exception:
            return None
        if bfv < 100:
            return None
        if bfv < 200:
            return BF_EDGE_COLOR["100-200"]
        return BF_EDGE_COLOR[">200"]

    def coord_of(year, region):
        if (region not in slot) or (year not in years):
            return None
        base_x = years.index(year) * YEAR_WIDTH
        which_row, col_idx = slot[region]
        if col_idx == "center":
            x = base_x + 2.0 * CELL_W
        else:
            x = base_x + subcols[col_idx]
        y = row_y[which_row]
        return x, y

    def draw_curved_arrow(
        ax, x0, y0, x1, y1,
        lw_inner, color_inner, color_outline,
        z=2.6,
        rad_sign_override=None
    ):
        pixel_dist = np.hypot(x1 - x0, y1 - y0)
        base_rad = CURVE_BASE + CURVE_GAIN * (pixel_dist / YEAR_WIDTH)

        rad_sign = 1.0 if (x1 - x0) >= 0 else -1.0

        if rad_sign_override is not None:
            rad_sign = float(rad_sign_override)

        rad = base_rad * rad_sign

        if color_outline is not None:
            ow = max(OUTLINE_MIN, lw_inner + OUTLINE_MARGIN)
            a1 = FancyArrowPatch(
                (x0, y0), (x1, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle='-|>',
                mutation_scale=ARROW_HEAD,
                linewidth=ow,
                color=color_outline, alpha=1.0, zorder=z,
                capstyle="round", joinstyle="round"
            )
            ax.add_patch(a1)

        a2 = FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle='-|>',
            mutation_scale=ARROW_HEAD,
            linewidth=lw_inner,
            color=color_inner, alpha=ARROW_ALPHA, zorder=z+0.1,
            capstyle="round", joinstyle="round"
        )
        ax.add_patch(a2)

    fig_w = 20
    fig_h = 13
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.30)

    LIGHT_GRAY, DARK_GRAY = YEAR_BAND_COLORS
    for j, yr in enumerate(years):
        x0 = j * YEAR_WIDTH
        x1 = x0 + YEAR_WIDTH
        face = LIGHT_GRAY if (j % 2 == 0) else DARK_GRAY
        ax.axvspan(x0, x1, facecolor=face, edgecolor="none", alpha=1.0, zorder=0.05)

    pts = {r: {"x": [], "y": []} for r in rows}

    for r in rows:
        for yr in years:
            xy = coord_of(yr, r)
            if xy is not None:
                x, y = xy
                pts[r]["x"].append(x)
                pts[r]["y"].append(y)

    for r in rows:
        ax.scatter(
            pts[r]["x"], pts[r]["y"],
            s=BUBBLE_PT**2,
            facecolors=col(r),
            edgecolors="black",
            linewidths=0.8,
            zorder=2.0,
            label=r
        )


    edges_plot = edges_df.copy()
    edges_plot = edges_plot[edges_plot["year_span"] <= MAX_YEAR_SPAN]
    print("BF>200:", (edges_plot["bayes_factor"] > 200).sum(),
          " 100<=BF<200:", ((edges_plot["bayes_factor"] >= 100) & (edges_plot["bayes_factor"] < 200)).sum())

    lw_map = make_lw_mapper(edges_plot)
    CONTINENTS = {"Americas", "Europe", "Asia", "Oceania"}
    for _, r in edges_plot.iterrows():
        sy, sreg = int(r["startYear"]), str(r["startRegion"])
        ey, ereg = int(r["endYear"]),   str(r["endRegion"])
        sxy = coord_of(sy, sreg)
        exy = coord_of(ey, ereg)
        if (sxy is None) or (exy is None):
            continue

        x0, y0 = sxy
        x1, y1 = exy
        lw = lw_map(float(r["mean_jumps"]))
        fill_col = EVIDENCE_COLOR.get(str(r["evidence"]), EVIDENCE_COLOR["NA"])
        edge_col = bf_edge_color(r.get("bayes_factor", None))
        if edge_col is None:
            continue

        src_is_elim = sreg in CONTINENTS
        src_is_end  = (sreg == "Endemic")
        dst_is_elim = ereg in CONTINENTS
        dst_is_end  = (ereg == "Endemic")

        if src_is_elim and dst_is_end:
            rad_sign = +1.0
        elif src_is_elim and dst_is_elim:
            rad_sign = -1.0
        elif src_is_end and dst_is_end:
            rad_sign = +1.0
        elif src_is_end and dst_is_elim:
            rad_sign = +1.0
        else:
            rad_sign = None

        draw_curved_arrow(
            ax, x0, y0, x1, y1,
            lw_inner=lw,
            color_inner=fill_col,
            color_outline=edge_col,
            z=2.6,
            rad_sign_override=rad_sign
        )


    ax.set_xticks(xticks_pos, [str(y) for y in years])
    ax.tick_params(axis="x", labelrotation=0)

    ax.set_yticks([row_y["continents"], row_y["endemic"]])
    ax.set_yticklabels(["Americas / Europe / Asia / Oceania", "Endemic"], fontsize=11)

    ax.set_xlim(-CELL_W * 0.5, len(years) * YEAR_WIDTH - YEAR_GAP + CELL_W * 0.5)

    y_vals = np.array(list(row_y.values()))
    y_pad  = 0.85 * CELL_H
    ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

    for s in ax.spines.values(): s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("black")

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, pad=6)
    ax.set_aspect("auto")
    ax.set_box_aspect(BOX_ASPECT)
    ax.set_xlabel("Year", labelpad=10)
    ax.set_title(title, pad=16)

    plt.tight_layout()
    fig.canvas.draw()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("Saved to:", save_path)
    plt.show()


def load_data():

    DATA_DIR_C1 = Path(__file__).resolve().parent
    FILE_DEME_C1 = DATA_DIR_C1 / "deme.list.csv"
    FILE_JUMP_C1 = DATA_DIR_C1 / "mj_his_merge.csv"
    FILE_BF_C1   = DATA_DIR_C1 / "Bayes.factor.test.result.csv"

    df_deme_C1 = pd.read_csv(FILE_DEME_C1)
    yr_reg_C1 = df_deme_C1["location"].apply(parse_location)
    presence_df_C1 = (
        pd.DataFrame(yr_reg_C1.tolist(), columns=["year","region"])
          .dropna()
          .drop_duplicates()
          .astype({"year": int, "region": str})
    )

    df_jump_C1 = pd.read_csv(FILE_JUMP_C1)

    df_jump_C1["time_num"] = pd.to_numeric(df_jump_C1["time"], errors="coerce")
    df_jump_C1 = df_jump_C1[df_jump_C1["time_num"].notna()].copy()

    df_jump_C1["date"] = df_jump_C1["time_num"].apply(date_decimal)
    df_jump_C1 = df_jump_C1[df_jump_C1["date"] >= pd.to_datetime("2011-01-01")].copy()
    df_jump_C1["year"] = df_jump_C1["date"].dt.year

    per_tree_C1 = (
        df_jump_C1.groupby(["treeId","year","startLocation","endLocation"], as_index=False)
                  .size()
                  .rename(columns={"size":"n_jumps"})
    )

    summary_C1 = (
        per_tree_C1.groupby(["year","startLocation","endLocation"], as_index=False)
                   .agg(mean_jumps=("n_jumps","mean"),
                        n_trees=("treeId","nunique"))
    )

    bf_raw_C1 = pd.read_csv(FILE_BF_C1)
    bf_clean_C1 = (
        bf_raw_C1.assign(
            start_name=lambda d: d["start_name"].astype(str).str.strip(),
            end_name=lambda d: d["end_name"].astype(str).str.strip(),
            bayes_factor=lambda d: pd.to_numeric(d["bayes_factor"], errors="coerce"),
            posterior_probability=lambda d: pd.to_numeric(d["posterior_probability"], errors="coerce"),
        )
        .dropna(subset=["start_name","end_name"])
        .groupby(["start_name","end_name"], as_index=False)
        .agg(bayes_factor=("bayes_factor","max"),
             posterior_probability=("posterior_probability","max"))
    )

    edges_df_C1 = (
        summary_C1.merge(bf_clean_C1,
                         left_on=["startLocation","endLocation"],
                         right_on=["start_name","end_name"], how="left")
                  .drop(columns=["start_name","end_name"])
    )

    edges_df_C1[["startYear","startRegion"]] = edges_df_C1["startLocation"].apply(
        lambda s: pd.Series(parse_location(s))
    )
    edges_df_C1[["endYear","endRegion"]] = edges_df_C1["endLocation"].apply(
        lambda s: pd.Series(parse_location(s))
    )
    edges_df_C1["evidence"] = edges_df_C1.apply(
        lambda r: evidence_code(r["startLocation"], r["endLocation"]), axis=1
    )

    edges_df_C1 = edges_df_C1[
        (pd.to_numeric(edges_df_C1["bayes_factor"], errors="coerce") >= BF_MIN_C1) &
        edges_df_C1["startYear"].notna() & edges_df_C1["endYear"].notna() &
        edges_df_C1["startRegion"].notna() & edges_df_C1["endRegion"].notna()
    ].copy()

    edges_df_C1["year_span"] = (edges_df_C1["endYear"] - edges_df_C1["startYear"]).abs()
    edges_df_C1 = edges_df_C1[edges_df_C1["year_span"] <= SPANNING_MAX_YEARS_C1].copy()
    years_C1 = list(range(2011, 2025 + 1))
    rows_C1  = ["Americas", "Europe", "Asia", "Oceania", "Endemic"]

    print(f"[OK][C1] presence_df_C1: {presence_df_C1.shape}, edges_df_C1 (BF≥{BF_MIN_C1}): {edges_df_C1.shape}")
    print(presence_df_C1.head())
    print(edges_df_C1.head())


    return presence_df_C1, edges_df_C1, years_C1, rows_C1, df_jump_C1
