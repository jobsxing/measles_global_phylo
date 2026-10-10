"""
Generate Panel F (chord diagram) and Panel G (monthly time series)
from mj_history_titration4.csv

Outputs (saved to SAVE_DIR):
  panel_F_chord_titration4.pdf / .png
  panel_G_timeseries_titration4.pdf / .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "4dta", "mj_history_titration4.csv")
SAVE_DIR = os.path.join(BASE_DIR, "figures") + os.sep
os.makedirs(SAVE_DIR, exist_ok=True)

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})
sns.set(style="white", context="talk")

STATES = ['EUR_eli', 'EUR_end', 'GLO_eli', 'GLO_end']

COLOR_MAP = {
    'GLO_end': '#a87680',
    'GLO_eli': '#bcb7b6',
    'EUR_end': '#637893',
    'EUR_eli': '#94a49d',
}

STATE_LABELS = {
    'EUR_eli': 'Europe\neliminated',
    'EUR_end': 'Europe\nendemic',
    'GLO_eli': 'Global\neliminated',
    'GLO_end': 'Global\nendemic',
}


print("Loading MJ data …")
mj_raw = pd.read_csv(DATA_PATH)
mj = mj_raw[mj_raw['startLocation'] != mj_raw['endLocation']].copy()

tree_ids = sorted(mj['treeId'].unique())
n_trees  = len(tree_ids)
print(f"  {n_trees} posterior trees, {len(mj):,} transitions")

pair_counts = (mj.groupby(['treeId', 'startLocation', 'endLocation'])
                 .size().reset_index(name='count'))

all_pairs = pd.DataFrame(
    [(t, s, e) for t in tree_ids for s in STATES for e in STATES if s != e],
    columns=['treeId', 'startLocation', 'endLocation']
)
pair_full = (all_pairs
             .merge(pair_counts, on=['treeId','startLocation','endLocation'], how='left')
             .fillna(0))
mean_mat_df = pair_full.groupby(['startLocation','endLocation'])['count'].mean().reset_index()

MAT = np.zeros((4, 4))
for _, row in mean_mat_df.iterrows():
    i = STATES.index(row['startLocation'])
    j = STATES.index(row['endLocation'])
    MAT[i, j] = row['count']

print("  Transition matrix (mean/tree):")
print(pd.DataFrame(MAT.round(1), index=STATES, columns=STATES))

def decimal_to_ym(t):
    year = int(t)
    month = int((t - year) * 12) + 1
    return f"{year}-{min(month, 12):02d}"

def hpd95(x):
    x = np.sort(x)
    n = len(x)
    w = int(np.floor(0.95 * n))
    if w == 0 or n - w == 0:
        return float(x[0]), float(x[-1])
    gaps = x[w:] - x[:n - w]
    idx  = int(gaps.argmin())
    return float(x[idx]), float(x[idx + w])

mj_win = mj[(mj['time'] >= 2018) & (mj['time'] < 2026.4)].copy()
mj_win['ym'] = mj_win['time'].apply(decimal_to_ym)
all_months = sorted(mj_win['ym'].unique())

print("\nBuilding monthly summary …")
records = []
for state in ['EUR_eli', 'EUR_end']:
    for direction, col in [('import', 'endLocation'), ('export', 'startLocation')]:
        filt     = mj_win[mj_win[col] == state]
        per_tree = filt.groupby(['treeId', 'ym']).size().reset_index(name='cnt')
        grid     = pd.DataFrame([(t, m) for t in tree_ids for m in all_months],
                                columns=['treeId', 'ym'])
        grid     = grid.merge(per_tree, on=['treeId','ym'], how='left').fillna(0)

        def summarize(g):
            vals = g['cnt'].values
            lo, hi = hpd95(vals)
            return pd.Series({'mean': vals.mean(), 'hpd_2_5': lo, 'hpd_97_5': hi})

        summ = grid.groupby('ym', group_keys=False).apply(summarize).reset_index()
        summ['state']     = state
        summ['direction'] = direction
        records.append(summ)

summary = pd.concat(records, ignore_index=True)
summary['time_mid'] = summary['ym'].apply(
    lambda ym: int(ym[:4]) + (int(ym[5:7]) - 0.5) / 12
)
summary = summary.sort_values(['state', 'direction', 'time_mid'])
print(f"  Monthly summary done: {len(summary)} rows")


def bezier_chord_path(a1s, a1e, a2s, a2e, r=0.72, n=200):
    """
    Filled Bezier ribbon between arc-segment 1 (a1s→a1e) and arc-segment 2 (a2s→a2e).
    The ribbon is bounded by:
      arc1 → Bezier(arc1_end → arc2_start) → arc2(reversed) → Bezier(arc2_end → arc1_start)
    This ensures the two Bezier edges are different, giving the ribbon visible width.
    """
    ctrl = np.array([0.0, 0.0])
    t    = np.linspace(0, 1, n)

    p1s = r * np.array([np.cos(a1s), np.sin(a1s)])
    p1e = r * np.array([np.cos(a1e), np.sin(a1e)])
    p2s = r * np.array([np.cos(a2s), np.sin(a2s)])
    p2e = r * np.array([np.cos(a2e), np.sin(a2e)])

    bx1 = (1-t)**2 * p1e[0] + 2*(1-t)*t * ctrl[0] + t**2 * p2s[0]
    by1 = (1-t)**2 * p1e[1] + 2*(1-t)*t * ctrl[1] + t**2 * p2s[1]

    bx2 = (1-t)**2 * p2e[0] + 2*(1-t)*t * ctrl[0] + t**2 * p1s[0]
    by2 = (1-t)**2 * p2e[1] + 2*(1-t)*t * ctrl[1] + t**2 * p1s[1]

    arc1 = np.linspace(a1s, a1e, 40)
    arc2 = np.linspace(a2s, a2e, 40)

    px = np.concatenate([r*np.cos(arc1), bx1, r*np.cos(arc2[::-1]), bx2])
    py = np.concatenate([r*np.sin(arc1), by1, r*np.sin(arc2[::-1]), by2])
    return px, py


def draw_panel_F(mat, states, colors, labels):
    n   = len(states)
    GAP = np.radians(6)

    involvement = mat.sum(axis=1) + mat.sum(axis=0)
    total        = involvement.sum()
    arc_span     = involvement / total * (2 * np.pi - n * GAP)

    CIRCLE_ORDER = [3, 2, 1, 0]

    start_angles = {}
    end_angles   = {}
    pos = np.radians(125)
    for idx in CIRCLE_ORDER:
        start_angles[idx] = pos
        end_angles[idx]   = pos + arc_span[idx]
        pos               = end_angles[idx] + GAP

    sub_out = {}
    sub_in  = {}
    for idx in CIRCLE_ORDER:
        outflows  = mat[idx, :]
        inflows   = mat[:, idx]
        out_total = outflows.sum()
        in_total  = inflows.sum()
        mid       = start_angles[idx] + arc_span[idx] * out_total / (out_total + in_total + 1e-9)

        pos2 = start_angles[idx]
        sub_out[idx] = {}
        for j in CIRCLE_ORDER:
            if j != idx and outflows[j] > 0:
                span = outflows[j] / (out_total + 1e-9) * (mid - start_angles[idx])
                sub_out[idx][j] = (pos2, pos2 + span)
                pos2 += span

        pos2 = mid
        sub_in[idx] = {}
        for j in CIRCLE_ORDER:
            if j != idx and inflows[j] > 0:
                span = inflows[j] / (in_total + 1e-9) * (end_angles[idx] - mid)
                sub_in[idx][j] = (pos2, pos2 + span)
                pos2 += span

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_aspect('equal')
    ax.axis('off')

    R_OUT  = 0.88
    R_IN   = 0.72
    R_LBL  = 1.06

    for idx in CIRCLE_ORDER:
        theta = np.linspace(start_angles[idx], end_angles[idx], 200)
        x_out = R_OUT * np.cos(theta);  y_out = R_OUT * np.sin(theta)
        x_in  = R_IN  * np.cos(theta[::-1]);  y_in = R_IN * np.sin(theta[::-1])
        ax.fill(np.concatenate([x_out, x_in]),
                np.concatenate([y_out, y_in]),
                color=colors[states[idx]], zorder=4, alpha=1.0)

    for i in CIRCLE_ORDER:
        for j in CIRCLE_ORDER:
            if i == j or mat[i, j] < 0.5:
                continue
            if j not in sub_out.get(i, {}) or i not in sub_in.get(j, {}):
                continue

            a1s, a1e = sub_out[i][j]
            a2s, a2e = sub_in[j][i]

            px, py = bezier_chord_path(a1s, a1e, a2s, a2e, r=R_IN)
            ax.fill(px, py, color=colors[states[i]], alpha=0.52, zorder=2,
                    linewidth=0.4, edgecolor=colors[states[i]])

            a2m  = (a2s + a2e) / 2
            tip  = np.array([R_IN * np.cos(a2m), R_IN * np.sin(a2m)])
            tail = tip * 0.78
            ax.annotate('',
                xy=tip, xytext=tail,
                arrowprops=dict(
                    arrowstyle='->', color=colors[states[i]],
                    lw=max(0.7, mat[i, j] / 65),
                    mutation_scale=10,
                ),
                zorder=6,
            )

    for idx in CIRCLE_ORDER:
        mid_angle = (start_angles[idx] + end_angles[idx]) / 2
        lx = R_LBL * np.cos(mid_angle)
        ly = R_LBL * np.sin(mid_angle)
        ha = 'center'
        if lx > 0.25:   ha = 'left'
        elif lx < -0.25: ha = 'right'
        ax.text(lx, ly, labels[states[idx]],
                ha=ha, va='center', fontsize=10,
                color=colors[states[idx]], fontweight='bold')

    ax.set_xlim(-1.40, 1.40)
    ax.set_ylim(-1.40, 1.40)
    plt.tight_layout()
    return fig


print("\nDrawing Panel F (chord diagram) …")
fig_F = draw_panel_F(MAT, STATES, COLOR_MAP, STATE_LABELS)
fig_F.savefig(SAVE_DIR + 'panel_F_chord_titration4.pdf', dpi=300, bbox_inches='tight')
fig_F.savefig(SAVE_DIR + 'panel_F_chord_titration4.png', dpi=300, bbox_inches='tight')
plt.close(fig_F)
print("  ✓ panel_F_chord_titration4.pdf / .png")



def draw_panel_G(summary):
    ticks, tick_labels = [], []
    for y in range(2018, 2027):
        for m, lab in [(6, 'Jun'), (12, 'Dec')]:
            t = y + (m - 0.5) / 12
            if 2018 <= t <= 2026.55:
                ticks.append(t)
                tick_labels.append(f"{lab}\n{y}")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axvspan(2018 + (2-1)/12,  2020 + (3-1)/12, color='#ededf7', zorder=0)
    ax.axvspan(2020 + (4-1)/12,  2023 + (4-1)/12, color='#fbe9e9', zorder=0)
    ax.axvspan(2023 + (5-1)/12,  2026.4, color='#f5f5f5', zorder=0)

    linestyle_map = {'import': '-', 'export': '--'}
    legend_labels = {
        ('EUR_eli', 'import'):  'Europe elimination import',
        ('EUR_eli', 'export'):  'Europe elimination export',
        ('EUR_end', 'import'):  'Europe endemic import',
        ('EUR_end', 'export'):  'Europe endemic export',
    }

    for state in ['EUR_eli', 'EUR_end']:
        for direction in ['import', 'export']:
            d = summary[(summary['state'] == state) & (summary['direction'] == direction)]
            if d.empty:
                continue
            ax.plot(d['time_mid'], d['mean'],
                    color=COLOR_MAP[state],
                    linestyle=linestyle_map[direction],
                    linewidth=1.1,
                    label=legend_labels[(state, direction)],
                    zorder=3)
            ax.fill_between(d['time_mid'], d['hpd_2_5'], d['hpd_97_5'],
                            color=COLOR_MAP[state], alpha=0.12, zorder=2)

    ax.set_xlim(2018, 2026.55)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9.5)
    ax.set_ylabel('Monthly number of Markov jump events', fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.tick_params(axis='both', which='major', length=0, width=0.3,
                   direction='out', labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    ax.legend(frameon=False, ncol=1, loc='upper right', fontsize=10)
    plt.tight_layout()
    return fig


print("Drawing Panel G (time series) …")
fig_G = draw_panel_G(summary)
fig_G.savefig(SAVE_DIR + 'panel_G_timeseries_titration4.pdf', dpi=300, bbox_inches='tight')
fig_G.savefig(SAVE_DIR + 'panel_G_timeseries_titration4.png', dpi=300, bbox_inches='tight')
plt.close(fig_G)
print("  ✓ panel_G_timeseries_titration4.pdf / .png")
print("\nAll done.")
