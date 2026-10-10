"""
Final N450 / WGS figure: temporal importation intensity, the surveillance
distribution behind it, and the bias that distribution creates.

Three stacked rows on one shared 2018-2026.5 time axis:
  a  global -> USA importation intensity inferred from each marker
  b  US sequences available per month in each dataset
  c  the resulting N450 : WGS ratio per 6-month bin

Row (a) uses the same numeric y-axis scale for both markers so their absolute
importation intensities can be compared directly.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Patch
from scipy import stats

from n450_wgs_import_concordance import (load_n450, load_wgs, load_sampling, rebin,
                                         rebin1, spearman, phase_randomise, NMONTH)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "N450-WGS-correlation")
TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "N450_WGS")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)
K = 6
N_SURR = 2000
RNG = np.random.default_rng(20260825)

LW_MEAN = 1.3
LW_BOUND = .3
FILL_ALPHA = .11
GRID = "#9A9A9A"

C_N450 = "#D9822B"
C_WGS = "#2E7D8F"
C_GAP = "#C0392B"
C_OK = "#4A3B6B"
C_EMPTY = "#ECECEC"


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r = stats.pearsonr(ex, ey)
    return float(r.statistic), float(r.pvalue)


def read_maxt_p():
    """Single source of truth: the scan-corrected p from the main analysis."""
    f = os.path.join(TABLES, "N450_WGS_concordance_statistics.csv")
    d = pd.read_csv(f)
    return float(d.loc[d.statistic == "selected_width_maxT_p", "value"].iloc[0])


def main():
    n3, W = load_n450("cap3"), load_wgs()
    An, Bn = rebin(n3, K), rebin(W, K)
    A, B = An.mean(0), Bn.mean(0)
    a_lo, a_hi = np.percentile(An, [2.5, 97.5], axis=0)
    b_lo, b_hi = np.percentile(Bn, [2.5, 97.5], axis=0)
    lr = np.log2(A / B)
    nb = len(A)
    edges = np.array([2018 + j * K / 12 for j in range(nb + 1)])
    mon_edges = np.array([2018 + i / 12 for i in range(NMONTH + 1)])

    s = load_sampling()
    wu, nu = s["wgs_usa"], s["n450_usa"]
    have = rebin1(wu, K) > 0
    c_ok = A[have].sum() / B[have].sum()
    c_gap = A[~have].sum() / B[~have].sum()
    c_all = A.sum() / B.sum()
    pr, pp = partial_spearman(rebin1(wu, K), lr, rebin1(nu, K))

    ia = RNG.integers(0, An.shape[0], 2000)
    ib = RNG.integers(0, Bn.shape[0], 2000)
    cp_ok = An[ia][:, have].sum(1) / np.maximum(Bn[ib][:, have].sum(1), 1e-9)
    cp_gap = An[ia][:, ~have].sum(1) / np.maximum(Bn[ib][:, ~have].sum(1), 1e-9)
    ok_lo, ok_hi = np.percentile(cp_ok, [2.5, 97.5])
    gap_lo, gap_hi = np.percentile(cp_gap, [2.5, 97.5])
    print(f"c both-see {c_ok:.2f} [{ok_lo:.2f}, {ok_hi:.2f}] | "
          f"WGS-blind {c_gap:.2f} [{gap_lo:.2f}, {gap_hi:.2f}]")

    rho = spearman(A, B)
    b_m = W.mean(0)
    null = np.array([spearman(A, np.clip(rebin1(phase_randomise(b_m, RNG), K), 0, None))
                     for _ in range(N_SURR)])
    null = null[np.isfinite(null)]
    p_sur = (np.sum(np.abs(null) >= abs(rho)) + 1) / (null.size + 1)
    p_maxt = read_maxt_p()

    print(f"rho={rho:.3f} surrogate p={p_sur:.4f} scan-corrected p={p_maxt:.4f}")
    print(f"c overall={c_all:.2f}  both-see={c_ok:.2f}  WGS-blind={c_gap:.2f}")
    print(f"partial r={pr:+.3f} p={pp:.4f}")

    plt.rcParams.update({"font.size": 8, "axes.linewidth": .7, "xtick.major.width": .7,
                         "ytick.major.width": .7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2.35, .62, 1.55], hspace=.36,
                  left=.108, right=.892, top=.948, bottom=.072)

    def shade(ax):
        for j in range(nb):
            if not have[j]:
                ax.axvspan(edges[j], edges[j + 1], color="#000000", alpha=.065, lw=0, zorder=0)

    def yeargrid(ax):
        for y in range(2019, 2027):
            ax.axvline(y, color=GRID, lw=.4, ls=(0, (3, 3)), alpha=.55, zorder=1)

    ax = fig.add_subplot(gs[0])
    shade(ax)
    yeargrid(ax)
    x = edges[:-1] + K / 24
    mkw = dict(lw=LW_MEAN, ls="--", dashes=(4.5, 2))
    ax.fill_between(x, a_lo, a_hi, color=C_N450, alpha=FILL_ALPHA, lw=0, zorder=2)
    ax.plot(x, a_lo, color=C_N450, lw=LW_BOUND, zorder=3)
    ax.plot(x, a_hi, color=C_N450, lw=LW_BOUND, zorder=3)
    ax.plot(x, A, color=C_N450, label="N450", zorder=4, **mkw)
    ax.set_ylabel(f"N450 importations / {K} months", color=C_N450, fontsize=8)
    ax.tick_params(axis="y", colors=C_N450)
    ax.spines["left"].set_color(C_N450)

    ax2 = ax.twinx()
    ax2.fill_between(x, b_lo, b_hi, color=C_WGS, alpha=FILL_ALPHA, lw=0, zorder=2)
    ax2.plot(x, b_lo, color=C_WGS, lw=LW_BOUND, zorder=3)
    ax2.plot(x, b_hi, color=C_WGS, lw=LW_BOUND, zorder=3)
    ax2.plot(x, B, color=C_WGS, label="whole genome", zorder=4, **mkw)
    ax2.set_ylabel(f"WGS importations / {K} months", color=C_WGS, fontsize=8)
    ax2.tick_params(axis="y", colors=C_WGS)
    ax2.spines["right"].set_color(C_WGS)
    ax2.spines["left"].set_color(C_N450)

    shared_top = max(a_hi.max(), b_hi.max()) * 1.06
    shared_ticks = mpl.ticker.MaxNLocator(nbins=8).tick_values(0, shared_top)
    shared_ticks = shared_ticks[shared_ticks >= 0]
    shared_top = float(shared_ticks[-1])
    ax.set_ylim(0, shared_top)
    ax2.set_ylim(0, shared_top)
    ax.set_yticks(shared_ticks)
    ax2.set_yticks(shared_ticks)
    ax.set_xlim(mon_edges[0], mon_edges[-1])
    ax.set_xticks(range(2018, 2027))
    ax.set_xticklabels([])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2 + [Patch(color="k", alpha=.11)],
              l1 + l2 + ["no US genome in the WGS dataset"],
              frameon=False, fontsize=7, loc="upper left", ncol=1,
              handlelength=2.2)
    ax.set_title("a   Global\u2192USA importation intensity", loc="left",
                 fontsize=8.8, fontweight="bold", pad=5)
    ax.text(.325, .955, f"N450 = {c_ok:.1f} \u00d7 WGS   (95% CrI {ok_lo:.1f}\u2013{ok_hi:.1f})\n"
                        f"Spearman $\\rho$ = {rho:.2f},  $p$ = {p_maxt:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
            linespacing=1.5)
    ax = fig.add_subplot(gs[1])
    vmax = np.log10(max(wu.max(), nu.max()) + 1)
    for row, (v, base) in enumerate(((wu, C_WGS), (nu, C_N450))):
        cmap = mpl.colors.LinearSegmentedColormap.from_list("m", ["#FFFFFF", base, "#101010"])
        for i in range(NMONTH):
            col = C_EMPTY if v[i] == 0 else cmap(.25 + .75 * np.log10(v[i] + 1) / vmax)
            ax.add_patch(Rectangle((mon_edges[i], -row - .43), 1 / 12, .86,
                                   facecolor=col, edgecolor="white", lw=.14))
    ax.set_xlim(mon_edges[0], mon_edges[-1])
    ax.set_ylim(-1.52, .52)
    ax.set_yticks([0, -1])
    ax.set_yticklabels([f"WGS\n{(wu>0).sum()}/{NMONTH} mo",
                        f"N450\n{(nu>0).sum()}/{NMONTH} mo"], fontsize=6.9)
    ax.set_xticks(range(2018, 2027))
    ax.set_xticklabels([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_title("b   US sequence availability", loc="left",
                 fontsize=8.8, fontweight="bold", pad=13)
    ax.text(0, 1.055, "grey = no sequence that month; shade = sequencing depth",
            transform=ax.transAxes, fontsize=6.9, color="#4A4A4A", va="bottom")

    ax = fig.add_subplot(gs[2])
    yeargrid(ax)
    for j in range(nb):
        ax.add_patch(Rectangle((edges[j], 0), K / 12, lr[j],
                               facecolor=C_GAP if not have[j] else C_OK,
                               edgecolor="white", lw=.5, zorder=2))
    ax.axhline(np.log2(c_gap), color=C_GAP, lw=1.3, ls="--", zorder=3,
               label=f"no US genome: {c_gap:.1f}\u00d7")
    ax.axhline(np.log2(c_ok), color=C_OK, lw=1.3, ls="-", zorder=3,
               label=f"US genomes: {c_ok:.1f}\u00d7")
    ax.set_xlim(mon_edges[0], mon_edges[-1])
    ax.set_ylim(0, lr.max() * 1.30)
    ax.set_xticks(range(2018, 2027))
    ax.set_xticklabels([str(y) for y in range(2018, 2027)], fontsize=7.5)
    ax.set_ylabel("$\\log_2$(N450 / WGS)", fontsize=8)
    ax.set_xlabel("Year", fontsize=8)
    leg = ax.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1.7,
                    title=f"partial $r$ = {pr:.2f}, $p$ = {pp:.3f}")
    leg.get_title().set_fontsize(6.9)
    ax.set_title("c   WGS under-detects without US genomes", loc="left",
                 fontsize=8.8, fontweight="bold", pad=5)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"N450_WGS_final_figure.{ext}"), dpi=400)
    plt.close(fig)

    pd.DataFrame(dict(
        bin_start=np.round(edges[:-1], 3),
        wgs_us_genomes=rebin1(wu, K).astype(int),
        wgs_us_months_covered=[int((wu[j * K:(j + 1) * K] > 0).sum()) for j in range(nb)],
        n450_us_seqs=rebin1(nu, K).astype(int),
        n450_us_months_covered=[int((nu[j * K:(j + 1) * K] > 0).sum()) for j in range(nb)],
        wgs_importations=np.round(B, 3), wgs_q025=np.round(b_lo, 3), wgs_q975=np.round(b_hi, 3),
        n450_importations=np.round(A, 3), n450_q025=np.round(a_lo, 3),
        n450_q975=np.round(a_hi, 3),
        log2_ratio=np.round(lr, 3),
        stratum=np.where(have, "US genomes present", "no US genome"),
    )).to_csv(os.path.join(TABLES, "N450_WGS_final_figure_data.csv"), index=False,
              encoding="utf-8-sig")

    pd.DataFrame([
        dict(statistic="spearman_rho", value=round(rho, 3),
             note=f"{K}-month bins, posterior-mean series"),
        dict(statistic="surrogate_p", value=round(p_sur, 4),
             note="phase-randomised monthly WGS series, re-binned"),
        dict(statistic="scan_corrected_maxT_p", value=round(p_maxt, 4),
             note="corrected for the bin-width/phase scan"),
        dict(statistic="c_overall", value=round(c_all, 3), note="ratio of totals"),
        dict(statistic="c_US_genomes_present", value=round(c_ok, 3),
             note=f"{int(have.sum())} bins"),
        dict(statistic="c_no_US_genome", value=round(c_gap, 3),
             note=f"{int((~have).sum())} bins"),
        dict(statistic="partial_r_ratio_vs_WGS_US_genomes", value=round(pr, 3),
             note=f"p={pp:.4f}, N450 US sampling controlled"),
        dict(statistic="WGS_US_genomes", value=int(wu.sum()),
             note=f"in {(wu>0).sum()}/{NMONTH} months"),
        dict(statistic="N450_US_sequences", value=int(nu.sum()),
             note=f"in {(nu>0).sum()}/{NMONTH} months"),
    ]).to_csv(os.path.join(TABLES, "N450_WGS_final_figure_statistics.csv"), index=False,
              encoding="utf-8-sig")
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
