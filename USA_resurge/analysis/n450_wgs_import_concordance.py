"""
Concordance / proportionality between N450- and WGS-inferred global -> USA
importation intensity, 2018-01 .. 2026-06.

Framing: N450(t) = c * WGS(t)  <=>  log N450(t) - log WGS(t) = log c is flat in t.
"Proportional" is therefore tested as constancy of the log-ratio, not as a high
correlation coefficient.

Bin width is scanned over k = 1..12 months and every phase offset. The primary
width is chosen by a PRE-SPECIFIED, OUTCOME-INDEPENDENT rule: the coarsest width
whose Spearman rho varies by less than PHASE_TOL across phase offsets, i.e. the
coarsest width at which the answer does not depend on where the bin grid is cut.
The argmax of rho is deliberately NOT used - rho rises monotonically with bin
width simply because coarsening concentrates the series into one trough-vs-peak
contrast, and at 12 months (8 bins) rho swings from 0.55 to 0.96 with phase.
The selected width is additionally reported with a maxT surrogate p-value that
corrects for having scanned the whole grid.

N450 cap = maximum sequences retained per state-month / deme-month stratum
(cap1 = 1190, cap2 = 1636, cap3 = 2015 sequences; WGS 2018+ = 2284). Lower caps
deliberately flatten epidemic intensity, so agreement with WGS is expected to
increase with cap; this is reported as a dose-response, not as robustness.

All statistics are propagated over random (N450 tree, WGS tree) posterior pairs.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

ROOT_N450 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "N450")
ROOT_WGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "final_core")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "N450-WGS-correlation")
TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "N450_WGS")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

N_PAIR = 2000
N_PAIR_SWEEP = 600
N_SURR = 2000
N_SURR_MAXT = 1000
K_GRID = [1, 2, 3, 4, 6, 8, 12]
PHASE_TOL = 0.10
OFFSET = 0.5
RNG = np.random.default_rng(20260825)

NMONTH = 102
MONTH_MID = np.array([2018.0 + (i + 0.5) / 12.0 for i in range(NMONTH)])
CAP_NSEQ = {"cap1": 1190, "cap2": 1636, "cap3": 2015}
WGS_NSEQ = 2284
SAMP_WGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WGS")
SAMP_N450 = ROOT_N450

C_N450 = "#D9822B"
C_WGS = "#2E7D8F"
C_NULL = "#B0B0B0"
C_PUR = "#4A3B6B"
C_PUR_L = "#7B68A6"
C_RED = "#C0392B"



def load_n450(cap="cap3"):
    """trees x 102 matrix of monthly global->USA Markov jumps."""
    f = os.path.join(ROOT_N450, "USA_extended_fig7_MJ_monthly.csv")
    d = pd.read_csv(f)
    d = d[(d.study == cap) & (d.direction == "Global to USA")]
    umon = np.sort(d.month_dec.unique())
    assert len(umon) == NMONTH, f"{cap}: {len(umon)} months, expected {NMONTH}"
    idx = {m: i for i, m in enumerate(umon)}
    d = d.assign(mi=d.month_dec.map(idx))
    m = (d.pivot_table(index="treeId", columns="mi", values="monthly",
                       aggfunc="sum", fill_value=0)
          .reindex(columns=range(NMONTH), fill_value=0))
    return m.to_numpy(float)


def load_wgs():
    f = os.path.join(ROOT_WGS, "globalUSA_temporal_MJ_by_tree_3month.csv")
    d = pd.read_csv(f)
    d = d[d.direction == "Global non-USA to USA"].copy()
    dt = pd.to_datetime(d.month)
    d["mi"] = (dt.dt.year - 2018) * 12 + (dt.dt.month - 1)
    d = d[(d.mi >= 0) & (d.mi < NMONTH)]
    m = (d.pivot_table(index="treeId", columns="mi", values="count",
                       aggfunc="sum", fill_value=0)
          .reindex(columns=range(NMONTH), fill_value=0))
    return m.to_numpy(float)


def load_sampling():
    """Monthly sequence counts behind each analysis, split USA vs non-USA.

    A nonUSA -> USA Markov jump can only be placed on a branch that has USA tips
    below it, so the number of USA sequences available in a period bounds how
    finely that analysis can resolve importations arriving in it.
    """
    tr = pd.read_csv(os.path.join(SAMP_WGS, "global_USA_2018plus_traits_2state.tsv"), sep="\t")
    dt = pd.read_csv(os.path.join(SAMP_WGS, "global_USA_2018plus_dates.tsv"), sep="\t")
    w = tr.merge(dt, left_on="traits", right_on="taxon")
    w["mi"] = ((w.decimal_date.astype(float) - 2018.0) * 12).astype(int)
    w = w[(w.mi >= 0) & (w.mi < NMONTH)]

    loc = pd.read_csv(os.path.join(
        SAMP_N450, "USA_state_month_plus_global_deme_month_cap3_location.txt"), sep="\t")
    tim = pd.read_csv(os.path.join(
        SAMP_N450, "USA_state_month_plus_global_deme_month_cap3_time.txt"), sep="\t")
    n = loc.merge(tim, on="name")
    d = pd.to_datetime(n.date)
    n["mi"] = (d.dt.year - 2018) * 12 + (d.dt.month - 1)
    n["state2"] = np.where(n.location == "USA", "USA", "nonUSA")
    n = n[(n.mi >= 0) & (n.mi < NMONTH)]

    def ser(df, col, val):
        v = np.zeros(NMONTH)
        c = df[df[col] == val].mi.value_counts()
        v[c.index.to_numpy()] = c.to_numpy()
        return v

    return dict(wgs_usa=ser(w, "2state", "USA"), wgs_non=ser(w, "2state", "nonUSA"),
                n450_usa=ser(n, "state2", "USA"), n450_non=ser(n, "state2", "nonUSA"))


def rebin(mat, k, phase=0):
    """Non-overlapping k-month bins starting at month `phase`; complete bins only."""
    nb = (NMONTH - phase) // k
    return mat[:, phase: phase + nb * k].reshape(mat.shape[0], nb, k).sum(2)


def rebin1(vec, k, phase=0):
    nb = (NMONTH - phase) // k
    return vec[phase: phase + nb * k].reshape(nb, k).sum(1)


def bin_mid(k, phase=0):
    nb = (NMONTH - phase) // k
    return np.array([2018.0 + (phase + (j + 0.5) * k) / 12.0 for j in range(nb)])



def ccc(x, y):
    """Lin's concordance correlation coefficient."""
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    den = vx + vy + (mx - my) ** 2
    return 2 * ((x - mx) * (y - my)).mean() / den if den > 0 else np.nan


def spearman(a, b):
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return float(stats.spearmanr(a, b).statistic)


def jsd(p, q):
    """Jensen-Shannon divergence (bits) between two normalised profiles."""
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        ok = a > 0
        return float(np.sum(a[ok] * np.log2(a[ok] / b[ok])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def sma_origin(x, y):
    return float(np.sqrt(np.sum(y ** 2) / np.sum(x ** 2)))


def ols_slope(t, y):
    t = t - t.mean()
    return float(np.sum(t * y) / np.sum(t ** 2))


def mann_kendall(y):
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return float(s), float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def phase_randomise(x, rng):
    """Surrogate preserving mean and power spectrum (hence autocorrelation)."""
    n = len(x)
    f = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, f.size)
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n) + x.mean()


def surrogate_p_binned(a_bin, b_month, k, stat, rng, nsurr=N_SURR):
    """Null built by phase-randomising the MONTHLY series, then re-binning.

    Randomising an already-binned 8- or 16-point series would destroy almost no
    structure; the autocorrelation that must be preserved lives at monthly scale.
    """
    obs = stat(a_bin, rebin1(b_month, k))
    null = np.array([stat(a_bin, np.clip(rebin1(phase_randomise(b_month, rng), k), 0, None))
                     for _ in range(nsurr)])
    null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1)
    return float(obs), float(p), null


def circular_shift_p(a_bin, b_month, k, stat, rng, nsurr=N_SURR):
    obs = stat(a_bin, rebin1(b_month, k))
    sh = rng.integers(1, NMONTH, nsurr)
    null = np.array([stat(a_bin, rebin1(np.roll(b_month, s), k)) for s in sh])
    null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1)
    return float(obs), float(p), null


def qi(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)), float(np.percentile(v, lo)), float(np.percentile(v, hi))


def fmt(t):
    return f"{t[0]:.3g} [{t[1]:.3g}, {t[2]:.3g}]"


def ecdf_from_counts(counts):
    c = np.cumsum(counts)
    return c / c[-1] if c[-1] > 0 else c


def wasserstein_time(cn, cw, mids):
    if cn.sum() <= 0 or cw.sum() <= 0:
        return np.nan
    return float(stats.wasserstein_distance(mids, mids, cn, cw))



def scan_bin_widths(nmat, wmat, rng):
    ia = rng.integers(0, nmat.shape[0], N_PAIR_SWEEP)
    ib = rng.integers(0, wmat.shape[0], N_PAIR_SWEEP)
    recs = []
    for k in K_GRID:
        for ph in range(k):
            A, B = rebin(nmat, k, ph), rebin(wmat, k, ph)
            if A.shape[1] < 5:
                continue
            a, b = A.mean(0), B.mean(0)
            pa = A / np.maximum(A.sum(1, keepdims=True), 1e-9)
            pb = B / np.maximum(B.sum(1, keepdims=True), 1e-9)
            rp = np.array([spearman(A[ia[i]], B[ib[i]]) for i in range(N_PAIR_SWEEP)])
            rp = rp[np.isfinite(rp)]
            cp = np.array([ccc(pa[ia[i]], pb[ib[i]]) for i in range(N_PAIR_SWEEP)])
            cp = cp[np.isfinite(cp)]
            recs.append(dict(k=k, phase=ph, nb=A.shape[1],
                             rho=spearman(a, b), ccc=ccc(a / a.sum(), b / b.sum()),
                             rho_lo=np.percentile(rp, 2.5), rho_med=np.median(rp),
                             rho_hi=np.percentile(rp, 97.5),
                             ccc_lo=np.percentile(cp, 2.5), ccc_med=np.median(cp)))
    d = pd.DataFrame(recs)
    agg = (d.groupby("k")
            .agg(nb=("nb", "median"),
                 rho_point_med=("rho", "median"), rho_point_min=("rho", "min"),
                 rho_point_max=("rho", "max"),
                 ccc_point_med=("ccc", "median"), ccc_point_min=("ccc", "min"),
                 ccc_point_max=("ccc", "max"),
                 crit_lo=("rho_lo", "median"), rho_pair_med=("rho_med", "median"))
            .reset_index())
    agg["phase_range"] = agg.rho_point_max - agg.rho_point_min
    return d, agg


def select_k(agg):
    """Coarsest width whose rho is phase-invariant to within PHASE_TOL."""
    ok = agg[agg.phase_range <= PHASE_TOL]
    return int(ok.k.max()) if len(ok) else int(agg.k.min())


def maxt_p(nmat, wmat, k_star, rng, nsurr=N_SURR_MAXT):
    """Null distribution of max-over-the-scanned-grid rho, for the selected width."""
    a_m, b_m = nmat.mean(0), wmat.mean(0)
    obs = float(np.median([spearman(rebin1(a_m, k_star, ph), rebin1(b_m, k_star, ph))
                           for ph in range(k_star)]))
    null = np.empty(nsurr)
    for s in range(nsurr):
        bs = phase_randomise(b_m, rng)
        best = -np.inf
        bs = np.clip(bs, 0, None)
        for k in K_GRID:
            vals = [abs(spearman(rebin1(a_m, k, ph), rebin1(bs, k, ph)))
                    for ph in range(k) if (NMONTH - ph) // k >= 5]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                best = max(best, float(np.median(vals)))
        null[s] = best
    return obs, float((np.sum(null >= abs(obs)) + 1) / (nsurr + 1)), null



def main():
    rows = []

    def rec(block, name, value, note=""):
        rows.append(dict(block=block, statistic=name, value=value, note=note))

    N = {cap: load_n450(cap) for cap in ("cap1", "cap2", "cap3")}
    W = load_wgs()
    n3 = N["cap3"]
    print(f"N450 cap3 {n3.shape}, WGS {W.shape}")

    for cap, m in N.items():
        rec("input", f"N450_{cap}_n_trees", m.shape[0], f"{CAP_NSEQ[cap]} sequences")
        rec("input", f"N450_{cap}_imports_per_tree_mean", round(m.sum(1).mean(), 2))
    rec("input", "WGS_n_trees", W.shape[0], f"{WGS_NSEQ} sequences")
    rec("input", "WGS_imports_per_tree_mean", round(W.sum(1).mean(), 2))
    rec("input", "window", "2018-01..2026-06", "102 months")

    print("scanning bin widths ...")
    sweep_full, agg = scan_bin_widths(n3, W, RNG)
    K = select_k(agg)
    print(agg.round(3).to_string(index=False))
    print(f"selected bin width: {K} months (phase-invariance rule, tol={PHASE_TOL})")
    for _, r in agg.iterrows():
        rec("bin_scan", f"k={int(r.k)}m_rho_phase_median", round(r.rho_point_med, 3),
            f"nbins={int(r.nb)}, phase range {r.phase_range:.3f} "
            f"[{r.rho_point_min:.3f}, {r.rho_point_max:.3f}]")
        rec("bin_scan", f"k={int(r.k)}m_CCC_phase_median", round(r.ccc_point_med, 3),
            f"phase range [{r.ccc_point_min:.3f}, {r.ccc_point_max:.3f}]")
    rec("bin_scan", "selection_rule", f"coarsest k with rho phase range <= {PHASE_TOL}",
        "pre-specified, outcome-independent; argmax of rho deliberately not used")
    rec("bin_scan", "selected_bin_width_months", K)
    rec("bin_scan", "argmax_rho_bin_width_months", int(agg.loc[agg.rho_point_med.idxmax(), "k"]),
        f"rho={agg.rho_point_med.max():.3f} but phase range "
        f"{agg.loc[agg.rho_point_med.idxmax(), 'phase_range']:.3f}; not used as primary")

    print("maxT null ...")
    maxt_obs, maxt_pval, maxt_null = maxt_p(n3, W, K, RNG)
    rec("bin_scan", "selected_width_rho_phase_median", round(maxt_obs, 3))
    rec("bin_scan", "selected_width_maxT_p", round(maxt_pval, 4),
        f"corrected for scanning k in {K_GRID} and all phase offsets")

    A, B = rebin(n3, K), rebin(W, K)
    mids = bin_mid(K)
    nb = A.shape[1]
    a_bar, b_bar = A.mean(0), B.mean(0)
    a_m, b_m = n3.mean(0), W.mean(0)
    rec("input", "primary_bin_months", K, f"{nb} bins, calendar-aligned (phase 0)")

    ia = RNG.integers(0, A.shape[0], N_PAIR)
    ib = RNG.integers(0, B.shape[0], N_PAIR)

    c_hat = a_bar.sum() / b_bar.sum()
    c_pairs = A[ia].sum(1) / np.maximum(B[ib].sum(1), 1e-9)
    c_q = qi(c_pairs)
    rec("proportionality", "c_overall_point", round(c_hat, 3), "ratio of posterior-mean totals")
    rec("proportionality", "c_overall_median_95CrI", fmt(c_q), "per tree pair")
    rec("proportionality", "SMA_slope_through_origin", round(sma_origin(b_bar, a_bar), 3))

    lr_bar = np.log2(a_bar / b_bar)
    LR = np.log2((A[ia] + OFFSET) / (B[ib] + OFFSET))
    lr_med = np.median(LR, 0)
    lr_lo, lr_hi = np.percentile(LR, [2.5, 97.5], axis=0)

    slope_bar = ols_slope(mids, lr_bar)
    slopes = np.array([ols_slope(mids, LR[i]) for i in range(N_PAIR)])
    p_drift = 2 * min((slopes > 0).mean(), (slopes < 0).mean())
    mk_s, mk_z, mk_p = mann_kendall(lr_bar)
    rec("proportionality", "log2ratio_mean_over_bins", round(lr_bar.mean(), 3))
    rec("proportionality", "log2ratio_monotone_drift_per_year", round(slope_bar, 4))
    rec("proportionality", "log2ratio_monotone_drift_median_95CrI", fmt(qi(slopes)))
    rec("proportionality", "log2ratio_monotone_drift_p", round(float(p_drift), 4),
        "NOT sufficient evidence of proportionality: blind to non-monotone excursions")
    rec("proportionality", "log2ratio_MannKendall_p", round(mk_p, 4))

    samp = load_sampling()
    s_wu, s_wn = rebin1(samp["wgs_usa"], K), rebin1(samp["wgs_non"], K)
    s_nu, s_nn = rebin1(samp["n450_usa"], K), rebin1(samp["n450_non"], K)
    have_usa = s_wu > 0
    rec("wgs_usa_sampling", "bins_with_zero_WGS_USA_genomes",
        f"{int((~have_usa).sum())}/{nb}")
    for lab, v in (("WGS_USA", samp["wgs_usa"]), ("WGS_nonUSA", samp["wgs_non"]),
                   ("N450_USA", samp["n450_usa"]), ("N450_nonUSA", samp["n450_non"])):
        rec("wgs_usa_sampling", f"{lab}_sequences_total", int(v.sum()),
            f"2018-19 {v[:24].sum():.0f} ({v[:24].sum()/v.sum():.1%}) | "
            f"2020-22 {v[24:60].sum():.0f} ({v[24:60].sum()/v.sum():.1%}) | "
            f"2023-24 {v[60:84].sum():.0f} ({v[60:84].sum()/v.sum():.1%}) | "
            f"2025-26 {v[84:].sum():.0f} ({v[84:].sum()/v.sum():.1%})")

    drivers = {}
    for lab, x in (("WGS USA sequences", s_wu), ("WGS non-USA sequences", s_wn),
                   ("N450 USA sequences", s_nu), ("N450 non-USA sequences", s_nn)):
        r = stats.spearmanr(x, lr_bar)
        drivers[lab] = (float(r.statistic), float(r.pvalue))
        rec("ratio_drivers", f"log2ratio_vs_{lab.replace(' ', '_')}",
            round(float(r.statistic), 3), f"Spearman p={r.pvalue:.4f}")

    per_masks = {
        "WGS has USA genomes": have_usa,
        "no WGS USA genome": ~have_usa,
    }
    periods = []
    for lab, msk in per_masks.items():
        cp = A[ia][:, msk].sum(1) / np.maximum(B[ib][:, msk].sum(1), 1e-9)
        q = qi(cp)
        periods.append(dict(label=lab, q=q, samples=cp, mask=msk,
                            n450=a_bar[msk].sum(), wgs=b_bar[msk].sum()))
        rec("stratified_by_USA_sampling", f"c_{lab}", fmt(q),
            f"{int(msk.sum())} bins; N450 {a_bar[msk].sum():.1f} vs "
            f"WGS {b_bar[msk].sum():.1f} per tree")
    p_period = float(np.mean(RNG.choice(periods[1]["samples"], N_PAIR)
                             > RNG.choice(periods[0]["samples"], N_PAIR)))
    rec("stratified_by_USA_sampling", "P(c_without_USA_genomes > c_with)",
        round(p_period, 4), "1.0 = the ratio is set by US genome availability")

    ba_x = 0.5 * (np.log2(a_bar) + np.log2(b_bar))
    bias, sd = float(lr_bar.mean()), float(lr_bar.std(ddof=1))
    loa = (bias - 1.96 * sd, bias + 1.96 * sd)
    ba_trend = stats.pearsonr(ba_x, lr_bar)
    rec("bland_altman", "bias_log2", round(bias, 3),
        f"geometric-mean ratio {2**bias:.2f}-fold (vs total-ratio c = {c_hat:.2f})")
    rec("bland_altman", "LoA_log2", f"[{loa[0]:.3f}, {loa[1]:.3f}]",
        f"fold [{2**loa[0]:.2f}, {2**loa[1]:.2f}]")
    rec("bland_altman", "proportional_bias_r", round(float(ba_trend.statistic), 3),
        f"p={ba_trend.pvalue:.4f}; negative = ratio inflated where counts are low")

    rho_obs, rho_p, rho_null = surrogate_p_binned(a_bar, b_m, K, spearman, RNG)
    _, rho_p_cs, _ = circular_shift_p(a_bar, b_m, K, spearman, RNG)

    def lg(u, v):
        return float(stats.pearsonr(np.log2(u + OFFSET), np.log2(v + OFFSET)).statistic)

    r_log, r_log_p, _ = surrogate_p_binned(a_bar, b_m, K, lg, RNG)

    pa = A / np.maximum(A.sum(1, keepdims=True), 1e-9)
    pb = B / np.maximum(B.sum(1, keepdims=True), 1e-9)
    pa_bar = a_bar / a_bar.sum()
    ccc_obs, ccc_p, _ = surrogate_p_binned(pa_bar, b_m, K,
                                           lambda u, v: ccc(u, v / v.sum()), RNG)

    rho_pairs = np.array([spearman(A[ia[i]], B[ib[i]]) for i in range(N_PAIR)])
    ccc_pairs = np.array([ccc(pa[ia[i]], pb[ib[i]]) for i in range(N_PAIR)])
    rec("association", "spearman_rho", round(rho_obs, 3),
        f"monthly phase-randomised p={rho_p:.4f}; circular-shift p={rho_p_cs:.4f}")
    rec("association", "spearman_rho_median_95CrI", fmt(qi(rho_pairs)), "per tree pair")
    rec("association", "pearson_r_log2", round(r_log, 3), f"surrogate p={r_log_p:.4f}")
    rec("association", "CCC_normalised_profiles", round(ccc_obs, 3), f"surrogate p={ccc_p:.4f}")
    rec("association", "CCC_median_95CrI", fmt(qi(ccc_pairs)), "per tree pair")

    ja = RNG.integers(0, A.shape[0], N_PAIR)
    jb = RNG.integers(0, B.shape[0], N_PAIR)
    d_between = np.array([jsd(pa[ia[i]], pb[ib[i]]) for i in range(N_PAIR)])
    d_wn = np.array([jsd(pa[ia[i]], pa[ja[i]]) for i in range(N_PAIR) if ia[i] != ja[i]])
    d_ww = np.array([jsd(pb[ib[i]], pb[jb[i]]) for i in range(N_PAIR) if ib[i] != jb[i]])
    p_excess = float((d_between[:, None] > RNG.choice(d_ww, (len(d_between), 1))).mean())
    rec("calibration", "JSD_between_markers_median_95CrI", fmt(qi(d_between)), "bits")
    rec("calibration", "JSD_within_N450_median_95CrI", fmt(qi(d_wn)), "bits")
    rec("calibration", "JSD_within_WGS_median_95CrI", fmt(qi(d_ww)), "bits")
    rec("calibration", "P(between > within_WGS)", round(p_excess, 3),
        "0.5 = indistinguishable from posterior noise; 1.0 = strict identity rejected")
    rec("calibration", "JSD_posterior_mean_profiles", round(jsd(pa_bar, b_bar / b_bar.sum()), 4))

    cdf_a = np.array([ecdf_from_counts(n3[i]) for i in range(n3.shape[0])])
    cdf_b = np.array([ecdf_from_counts(W[i]) for i in range(W.shape[0])])
    ks_between = np.array([np.max(np.abs(cdf_a[ia[i]] - cdf_b[ib[i]])) for i in range(N_PAIR)])
    ks_wn = np.array([np.max(np.abs(cdf_a[ia[i]] - cdf_a[ja[i]])) for i in range(N_PAIR)])
    ks_ww = np.array([np.max(np.abs(cdf_b[ib[i]] - cdf_b[jb[i]])) for i in range(N_PAIR)])
    ws_between = np.array([wasserstein_time(n3[ia[i]], W[ib[i]], MONTH_MID) for i in range(N_PAIR)])
    ws_wn = np.array([wasserstein_time(n3[ia[i]], n3[ja[i]], MONTH_MID) for i in range(N_PAIR)])
    rec("invasion_time", "KS_between_markers_median_95CrI", fmt(qi(ks_between)))
    rec("invasion_time", "KS_within_N450_median_95CrI", fmt(qi(ks_wn)))
    rec("invasion_time", "KS_within_WGS_median_95CrI", fmt(qi(ks_ww)))
    rec("invasion_time", "Wasserstein_between_markers_years", fmt(qi(ws_between)))
    rec("invasion_time", "Wasserstein_within_N450_years", fmt(qi(ws_wn)))
    med_a, med_b = np.median(cdf_a, 0), np.median(cdf_b, 0)
    rec("invasion_time", "KS_posterior_median_ECDFs", round(float(np.max(np.abs(med_a - med_b))), 4))

    lags = np.arange(-12, 13)

    def ccf(x, y):
        out, n = [], len(x)
        for L in lags:
            xs, ys = x[max(0, L): n + min(0, L)], y[max(0, -L): n + min(0, -L)]
            out.append(np.corrcoef(xs, ys)[0, 1] if xs.std() > 0 and ys.std() > 0 else np.nan)
        return np.array(out)

    ccf_bar = ccf(a_m, b_m)
    CCF = np.array([ccf(n3[ia[i]], W[ib[i]]) for i in range(500)])
    ccf_lo, ccf_hi = np.nanpercentile(CCF, [2.5, 97.5], axis=0)
    peak = int(lags[np.nanargmax(ccf_bar)])
    peak_pairs = lags[np.nanargmax(CCF, 1)]
    rec("timing", "CCF_peak_lag_months", peak, "positive = N450 later than WGS")
    rec("timing", "CCF_peak_lag_median_95CrI", fmt(qi(peak_pairs)), "per tree pair")
    rec("timing", "CCF_at_lag0", round(float(ccf_bar[lags == 0][0]), 3))
    rec("timing", "CCF_at_peak", round(float(np.nanmax(ccf_bar)), 3))

    caps = []
    for cap, m in N.items():
        Ak = rebin(m, K)
        ic = RNG.integers(0, Ak.shape[0], N_PAIR)
        pak = Ak / np.maximum(Ak.sum(1, keepdims=True), 1e-9)
        rr = np.array([spearman(Ak[ic[i]], B[ib[i]]) for i in range(N_PAIR)])
        cc = np.array([ccc(pak[ic[i]], pb[ib[i]]) for i in range(N_PAIR)])
        cs = Ak[ic].sum(1) / np.maximum(B[ib].sum(1), 1e-9)
        caps.append(dict(cap=cap, nseq=CAP_NSEQ[cap], rho=qi(rr), cccv=qi(cc), c=qi(cs),
                         rho_point=spearman(Ak.mean(0), b_bar)))
        rec("cap_dose_response", f"{cap}_spearman_median_95CrI", fmt(qi(rr)),
            f"{CAP_NSEQ[cap]} sequences")
        rec("cap_dose_response", f"{cap}_CCC_median_95CrI", fmt(qi(cc)))
        rec("cap_dose_response", f"{cap}_c_median_95CrI", fmt(qi(cs)))
        rec("cap_dose_response", f"{cap}_trough_share_of_importations",
            f"{m.mean(0)[24:60].sum() / m.mean(0).sum():.1%}", "2020-2022")
    rec("cap_dose_response", "WGS_trough_share_of_importations",
        f"{b_m[24:60].sum() / b_m.sum():.1%}", "2020-2022")

    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "N450_WGS_concordance_statistics.csv"),
                              index=False, encoding="utf-8-sig")
    sweep_full.to_csv(os.path.join(TABLES, "N450_WGS_bin_width_scan.csv"), index=False,
                      encoding="utf-8-sig")

    plt.rcParams.update({"font.size": 8, "axes.linewidth": .7, "xtick.major.width": .7,
                         "ytick.major.width": .7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(13.8, 8.6))
    gs = GridSpec(2, 3, figure=fig, hspace=.46, wspace=.36,
                  left=.088, right=.982, top=.905, bottom=.075)

    half = K / 24.0
    no_usa_spans = [(mids[j] - half, mids[j] + half) for j in range(nb) if not have_usa[j]]

    def shade_gap(ax):
        for lo, hi in no_usa_spans:
            ax.axvspan(lo, hi, color="#000000", alpha=.075, lw=0, zorder=0)

    ax = fig.add_subplot(gs[0, 0])
    shade_gap(ax)
    a_lo, a_hi = np.percentile(A, [2.5, 97.5], axis=0)
    b_lo, b_hi = np.percentile(B, [2.5, 97.5], axis=0)
    ax.fill_between(mids, a_lo, a_hi, color=C_N450, alpha=.20, lw=0)
    ax.plot(mids, a_bar, color=C_N450, lw=1.8, label="N450 (cap3)")
    ax.set_ylabel(f"N450 importations / {K} months", color=C_N450)
    ax.tick_params(axis="y", colors=C_N450)
    ax2 = ax.twinx()
    ax2.fill_between(mids, b_lo, b_hi, color=C_WGS, alpha=.20, lw=0)
    ax2.plot(mids, b_bar, color=C_WGS, lw=1.8, label="WGS")
    ax2.set_ylabel(f"WGS importations / {K} months", color=C_WGS)
    ax2.tick_params(axis="y", colors=C_WGS)
    c_lock = a_bar[have_usa].sum() / b_bar[have_usa].sum()
    top = max(a_hi.max(), (b_hi * c_lock).max()) * 1.05
    ax.set_ylim(0, top)
    ax2.set_ylim(0, top / c_lock)
    ax.set_xlabel("Year")
    ax.set_title(f"a  Importation intensity, axes locked at c = {c_lock:.2f}\n"
                 f"    (ratio where both markers have US sequences)",
                 loc="left", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2 + [mpl.patches.Patch(color="k", alpha=.12)],
              l1 + l2 + ["no WGS USA genome in bin"], frameon=False, fontsize=7,
              loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(b_bar, a_bar, c=mids, cmap="viridis", s=48, zorder=3,
                    edgecolor="k", linewidth=.4)
    xx = np.linspace(0, b_bar.max() * 1.08, 50)
    ax.fill_between(xx, c_q[1] * xx, c_q[2] * xx, color="k", alpha=.10, lw=0, zorder=1,
                    label="95% CrI")
    ax.plot(xx, c_hat * xx, color="k", lw=1.3, zorder=2, label=f"proportional fit, c = {c_hat:.2f}")
    ax.set_xlabel(f"WGS importations / {K} months")
    ax.set_ylabel(f"N450 importations / {K} months")
    ax.set_xlim(0, b_bar.max() * 1.08)
    ax.set_ylim(0, a_bar.max() * 1.08)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(f"b  $\\rho$ = {rho_obs:.2f} (surrogate p = {rho_p:.3f}; "
                 f"scan-corrected p = {maxt_pval:.3f})", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, pad=.02, fraction=.045)
    cb.set_label("Year", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[0, 2])
    shade_gap(ax)
    ax.fill_between(mids, lr_lo, lr_hi, color=C_PUR_L, alpha=.20, lw=0, label="95% CrI")
    ax.plot(mids, lr_med, color=C_PUR_L, lw=1.0, ls="--", label="posterior median")
    ax.plot(mids, lr_bar, "o-", color=C_PUR, lw=1.6, ms=4, label="posterior mean series")
    ax.axhline(np.log2(c_hat), color="k", lw=1.1, ls=":",
               label=f"constant ratio $\\log_2 c$ = {np.log2(c_hat):.2f}")
    ax.set_xlabel("Year")
    ax.set_ylabel("$\\log_2$(N450 / WGS)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_title("c  Ratio is not constant; it tracks US genome\n"
                 f"    availability, not calendar time", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    rho_d, p_d = drivers["WGS USA sequences"]
    rho_g, p_g = drivers["WGS non-USA sequences"]
    x = np.log10(s_wu + 1)
    ax.scatter(x, lr_bar, c=mids, cmap="viridis", s=52, edgecolor="k", linewidth=.4, zorder=3)
    for p, col, ls in ((periods[1], C_RED, "--"), (periods[0], C_PUR, "-")):
        ax.axhline(np.log2(p["q"][0]), color=col, lw=1.3, ls=ls,
                   label=f"{p['label']}: c = {p['q'][0]:.1f}")
    ax.set_xlabel("$\\log_{10}$(WGS USA genomes in bin + 1)")
    ax.set_ylabel("$\\log_2$(N450 / WGS)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title(f"d  Driver: WGS USA genomes $\\rho$ = {rho_d:+.2f} (p = {p_d:.3f});\n"
                 f"    WGS non-USA genomes $\\rho$ = {rho_g:+.2f} (p = {p_g:.2f}, n.s.)",
                 loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    shade_gap(ax)
    for cdf, col, lab in ((cdf_a, C_N450, "N450 (cap3)"), (cdf_b, C_WGS, "WGS")):
        lo, hi = np.percentile(cdf, [2.5, 97.5], axis=0)
        ax.fill_between(MONTH_MID, lo, hi, color=col, alpha=.22, lw=0)
        ax.plot(MONTH_MID, np.median(cdf, 0), color=col, lw=1.8, label=lab)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative share of importations")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ks_q, kw_q = qi(ks_between), qi(ks_ww)
    ax.set_title(f"e  Invasion-time distribution: KS between = {ks_q[0]:.2f}\n"
                 f"    [{ks_q[1]:.2f}, {ks_q[2]:.2f}] vs within WGS = {kw_q[0]:.2f}",
                 loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(0, max(np.percentile(d_between, 99.5), np.percentile(d_wn, 99.5),
                              np.percentile(d_ww, 99.5)), 55)
    ax.hist(d_wn, bins=bins, color=C_N450, alpha=.55, density=True,
            label="within N450 (tree vs tree)")
    ax.hist(d_ww, bins=bins, color=C_WGS, alpha=.55, density=True,
            label="within WGS (tree vs tree)")
    ax.hist(d_between, bins=bins, histtype="step", color="k", lw=1.6, density=True,
            label="between markers")
    ax.set_xlabel("Jensen-Shannon divergence of temporal profile (bits)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title(f"f  Strict profile identity rejected:\n"
                 f"    P(between > within WGS) = {p_excess:.2f}", loc="left", fontweight="bold")

    fig.suptitle("Global\u2192USA importation intensity inferred from N450 vs whole genomes, "
                 f"2018\u20132026.5 ({K}-month bins, N450 cap3)",
                 fontsize=10.5, fontweight="bold", x=.055, ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"N450_WGS_concordance_main.{ext}"), dpi=400)
    plt.close(fig)

    fig = plt.figure(figsize=(13.4, 8.0))
    gs = GridSpec(2, 3, figure=fig, hspace=.46, wspace=.30,
                  left=.06, right=.985, top=.905, bottom=.085)

    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(agg.k, agg.rho_point_min, agg.rho_point_max, color=C_PUR, alpha=.16,
                    lw=0, label="range across phase offsets")
    ax.plot(agg.k, agg.rho_point_med, "o-", color=C_PUR, lw=1.7, ms=5,
            label="Spearman $\\rho$ (phase median)")
    ax.plot(agg.k, agg.ccc_point_med, "s-", color=C_WGS, lw=1.5, ms=5, label="Lin's CCC")
    ax.axvline(K, color=C_RED, lw=1.3, ls="--", label=f"selected: {K} months")
    for _, r in agg.iterrows():
        ax.annotate(f"{int(r.nb)}", (r.k, r.rho_point_max), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6, color=C_PUR)
    ax.set_xscale("log")
    ax.set_xticks(K_GRID)
    ax.set_xticklabels([str(k) for k in K_GRID])
    ax.set_xlabel("Bin width (months); labels = number of bins")
    ax.set_ylabel("Statistic")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("a  $\\rho$ rises with bin width - but so does\n    phase fragility",
                 loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(len(agg)), agg.phase_range, color=[C_RED if pr > PHASE_TOL else C_PUR
                                                    for pr in agg.phase_range], width=.6)
    ax.axhline(PHASE_TOL, color="k", lw=1.1, ls="--",
               label=f"pre-specified tolerance = {PHASE_TOL}")
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([str(int(k)) for k in agg.k])
    ax.set_xlabel("Bin width (months)")
    ax.set_ylabel("Range of $\\rho$ across phase offsets")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(f"b  Selection rule: coarsest width below the line\n    \u2192 {K} months",
                 loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(maxt_null, bins=50, color=C_NULL, density=True,
            label="null: max $\\rho$ over the whole scan")
    ax.axvline(maxt_obs, color=C_RED, lw=1.8, label=f"observed at {K} months = {maxt_obs:.2f}")
    ax.set_xlabel("Spearman $\\rho$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(f"c  Scan-corrected: maxT p = {maxt_pval:.4f}", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    xs = [c["nseq"] for c in caps]
    for key, col, mk, lab in (("rho", C_PUR, "o", "Spearman $\\rho$"),
                              ("cccv", C_WGS, "s", "Lin's CCC")):
        med = [c[key][0] for c in caps]
        lo = [c[key][0] - c[key][1] for c in caps]
        hi = [c[key][2] - c[key][0] for c in caps]
        ax.errorbar(xs, med, yerr=[lo, hi], fmt=mk + "-", color=col, lw=1.5, ms=6,
                    capsize=3, label=lab)
    ax.axhline(0, color="k", lw=.6)
    ax.axvline(WGS_NSEQ, color=C_RED, lw=1.1, ls=":", label=f"WGS n = {WGS_NSEQ}")
    for c in caps:
        ax.annotate(c["cap"], (c["nseq"], c["rho"][2]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color=C_PUR)
    ax.set_xlabel("N450 sequences retained (cap = max per state-month)")
    ax.set_ylabel("Agreement with WGS (median, 95% CrI)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("d  Agreement increases with sampling depth,\n    as expected by construction",
                 loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(lags, ccf_lo, ccf_hi, color=C_PUR_L, alpha=.22, lw=0, label="95% CrI")
    ax.plot(lags, ccf_bar, "o-", color=C_PUR, lw=1.6, ms=4, label="posterior mean series")
    ax.axvline(0, color="k", lw=.8, ls=":")
    ax.axvline(peak, color=C_RED, lw=1.1, ls="--", label=f"peak at {peak:+d} months")
    ax.set_xlabel("Lag (months); positive = N450 later than WGS")
    ax.set_ylabel("Cross-correlation")
    ax.legend(frameon=False, fontsize=7, loc="lower center")
    ax.set_title("e  No resolvable systematic timing offset", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    dat = [rho_pairs[np.isfinite(rho_pairs)], ccc_pairs[np.isfinite(ccc_pairs)],
           np.log2(c_pairs), slopes]
    labs = ["Spearman $\\rho$", "Lin's CCC", "$\\log_2 c$", "drift slope\n(per year)"]
    vp = ax.violinplot(dat, showmedians=True, widths=.8)
    for b in vp["bodies"]:
        b.set_facecolor(C_PUR_L)
        b.set_alpha(.55)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        vp[key].set_color(C_PUR)
        vp[key].set_linewidth(1.1)
    ax.axhline(0, color="k", lw=.6, ls=":")
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("Value")
    ax.set_title("f  Posterior distributions over tree pairs", loc="left", fontweight="bold")

    fig.suptitle("Bin-width selection, scan correction and sampling-depth dependence of the "
                 "N450\u2013WGS importation concordance",
                 fontsize=10.5, fontweight="bold", x=.06, ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"N450_WGS_concordance_sensitivity.{ext}"), dpi=400)
    plt.close(fig)

    print("\n".join(f"{r['block']:18s} {r['statistic']:48s} {r['value']}  {r['note']}"
                    for r in rows))
    print("\nwrote ->", OUT)


if __name__ == "__main__":
    main()
