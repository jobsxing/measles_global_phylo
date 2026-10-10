"""Five-group temporal and source/sink comparison, post-2018 common window.

Groups (all five are now 13-deme BSSVS discrete trait analyses)
  WGS        Matched-taxon WGS                     n=901
  N450       Matched-taxon N450, uncalibrated      n=901
  Calibrated Matched-taxon N450, root-calibrated   n=901
  Main       Main N450, uncalibrated               n=3794
  MainCal    Main N450, root-calibrated            n=3794

`Main` is the re-run BSSVS analysis in revision_GLM/combined, not the earlier
VI/GLM chain. That removes the model confound in the previous version of this
figure, where Main was GLM and MainCal was BSSVS, so the main-arm difference
mixed calibration with a change of migration model.

Bayes factor filter: only routes with BF > BF_MIN in the *same* analysis
contribute jumps. The threshold is set on the reference analysis (WGS), where
BF > 3 discards 125 of the 156 directed routes while retaining ~96% of the
post-2018 events. Each analysis is then filtered by its own support, so a
better-powered dataset is not penalised for resolving more routes, and within
each calibrated/uncalibrated pair the taxon set is identical.

Panel A pairs each marker with its calibrated counterpart in one axis, so the
question "did calibrating the root move the temporal signal?" is read within a
panel rather than across panels:
  A1 WGS alone (the reference timescale)
  A2 N450 901  calibrated vs uncalibrated
  A3 N450 3794 calibrated vs uncalibrated
Panels B and C carry all five series.

Requires tables/30_bf_all_groups.tsv from prep_bf.py.
Reads only; writes figures and tables next to the existing four-group outputs.
"""
from __future__ import annotations

import sys
import argparse
import textwrap
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arviz as az

ROOT = Path(__file__).resolve().parent
OUT = ROOT
PUB = ROOT
BASE = ROOT
RAW = ROOT
CAL = ROOT
CAL3794 = ROOT
UNCAL3794 = ROOT
TABLES = ROOT
FIGURES = ROOT
BF_MIN = 3.0
SHOW_FOOTNOTE = False
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
import analysis_helpers as old

KEYS = ['WGS', 'N450', 'Calibrated', 'Main', 'MainCal']
BIG = {'Main', 'MainCal'}
LABEL = {
    'WGS': 'Matched-taxon WGS (n=901)',
    'N450': 'Matched-taxon N450, uncalibrated (n=901)',
    'Calibrated': 'Matched-taxon N450, age calibrated (n=901)',
    'Main': 'Main N450, uncalibrated (n=3794)',
    'MainCal': 'Main N450, age calibrated (n=3794)',
}
COLOR = {'WGS': '#245A78', 'N450': '#C34C59', 'Calibrated': '#268574',
         'Main': '#A08232', 'MainCal': '#6A4C93'}
XML_DIR = ROOT.parent / 'xml'
XML = {
    'WGS': XML_DIR / 'WGS' / 'geo_13deme_WGS_rep_1.xml',
    'N450': XML_DIR / 'N450' / 'geo_13deme_N450_rep_1.xml',
    'Calibrated': XML_DIR / 'N450_901_root_prior_change' / 'geo13cal_rep_1.xml',
    'Main': XML_DIR / 'N450_3794_root_prior_change' / 'geo13cal3794_rep_1.xml',
    'MainCal': XML_DIR / 'N450_3794_root_prior_change' / 'geo13cal3794_rep_1.xml',
}
HISTORY = {
    'WGS': ROOT / 'geo_13deme_WGS_sub100.deme13.MJhistory.txt',
    'N450': ROOT / 'geo_13deme_N450_sub100.deme13.MJhistory.txt',
    'Calibrated': ROOT / 'geo13cal_MJhistory.txt',
    'Main': ROOT / 'bssvs13_uncal_MJhistory.txt',
    'MainCal': ROOT / 'geo13cal3794_MJhistory.txt',
}

STATES = old.STATE_ORDER[13]
PANELS = [('A1', ['WGS']), ('A2', ['N450', 'Calibrated']), ('A3', ['Main', 'MainCal'])]
PANEL_TITLE = {'A1': 'Matched-taxon WGS (n=901)',
               'A2': 'Matched-taxon N450 (age calibrated vs. uncalibrated; n=901)',
               'A3': 'Main N450 (age calibrated vs. uncalibrated; n=3794)'}
FS = 2.0
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11 * FS,
                     'axes.titlesize': 14 * FS, 'axes.labelsize': 12 * FS,
                     'xtick.labelsize': 10 * FS, 'ytick.labelsize': 11 * FS,
                     'legend.fontsize': 10 * FS, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none'})


def summary(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(mean=np.nan, median=np.nan, lo95=np.nan, hi95=np.nan,
                    lo50=np.nan, hi50=np.nan, n=0)
    iv = {p: az.hdi(a, hdi_prob=p) if len(a) > 1 else [a[0], a[0]] for p in (.5, .95)}
    return dict(mean=a.mean(), median=np.median(a), lo95=iv[.95][0], hi95=iv[.95][1],
                lo50=iv[.5][0], hi50=iv[.5][1], n=len(a))


def xml_metadata(path):
    root = ET.parse(path).getroot()
    rows = []
    for t in root.findall('./taxa/taxon'):
        attrs = {a.attrib['name']: (a.text or '').strip() for a in t.findall('attr')}
        state = attrs.get('deme13', attrs.get('geography'))
        rows.append({'taxon': t.attrib['id'], 'date': float(t.find('date').attrib['value']),
                     'deme': old.VI_STATE_MAP.get(state, state)})
    return pd.DataFrame(rows).set_index('taxon')


def bf_masks():
    """Supported route sets, one per analysis, at BF > BF_MIN."""
    bf = pd.read_csv(TABLES / '30_bf_all_groups.tsv', sep='\t')
    assert set(bf.dataset) == set(KEYS), 'BF table does not cover the five groups'
    masks, rows = {}, []
    ref = set(map(tuple, bf.loc[(bf.dataset == 'WGS') & (bf.bf > BF_MIN),
                                ['source', 'sink']].to_numpy()))
    for k in KEYS:
        q = bf[bf.dataset == k]
        assert len(q) == 156, f'{k}: {len(q)} routes in the BF table'
        masks[k] = set(map(tuple, q.loc[q.bf > BF_MIN, ['source', 'sink']].to_numpy()))
        assert masks[k], f'{k}: no route passes BF > {BF_MIN}'
    for (s, t) in [(s, t) for s in STATES for t in STATES if s != t]:
        r = {'source': s, 'sink': t, 'in_WGS_reference_set': (s, t) in ref}
        for k in KEYS:
            r[f'BF_{k}'] = float(bf.loc[(bf.dataset == k) & (bf.source == s)
                                        & (bf.sink == t), 'bf'].iloc[0])
            r[f'supported_{k}'] = (s, t) in masks[k]
        rows.append(r)
    pd.DataFrame(rows).to_csv(TABLES / '31_route_bf_mask.tsv', sep='\t', index=False)
    return masks, ref


def apply_mask(h, keep):
    ok = pd.MultiIndex.from_arrays([h.source, h.sink]).isin(list(keep))
    return h[ok].copy()


def counts(h, start, end):
    ids = h.treeId.drop_duplicates().tolist()
    x = h[(h.time >= start) & (h.time <= end)]
    idx = pd.MultiIndex.from_product([STATES, STATES], names=['source', 'sink'])
    a = (x.groupby(['treeId', 'source', 'sink']).size().unstack(['source', 'sink'])
         .reindex(index=ids, columns=idx).fillna(0).to_numpy()).reshape(len(ids), 13, 13)
    assert int(a.sum()) == len(x), 'history events did not reconcile'
    assert np.trace(a, axis1=1, axis2=2).sum() == 0, 'self transitions present'
    return a


def main():
    meta = {k: xml_metadata(XML[k]) for k in KEYS}
    for k in KEYS:
        want = 3794 if k in BIG else 901
        assert len(meta[k]) == want, f'{k}: {len(meta[k])} taxa, expected {want}'
        assert set(meta[k].deme) <= set(STATES), f'{k}: unexpected demes'
    start, end = 2018.0, min(m.date.max() for m in meta.values())
    print(f'common window: {start} .. {end:.4f}')

    hist = {k: old.read_history(HISTORY[k], k, vi=(k in BIG)) for k in KEYS}
    for k in KEYS:
        assert hist[k].time.max() <= meta[k].date.max() + 1e-6, f'{k}: event after last sample'

    masks, ref = bf_masks()
    print(f'\nBayes factor filter, BF > {BF_MIN:g} within each analysis '
          f'(threshold set on WGS: {len(ref)}/156 routes)')
    audit = []
    for k in KEYS:
        w = hist[k][hist[k].time >= start]
        kept = apply_mask(w, masks[k])
        by_ref = apply_mask(w, ref)
        audit.append({'dataset': k, 'bf_threshold': BF_MIN,
                      'routes_supported': len(masks[k]),
                      'routes_in_WGS_reference_set': len(ref),
                      'post2018_events_all': len(w),
                      'post2018_events_kept': len(kept),
                      'frac_events_kept': len(kept) / len(w),
                      'frac_events_kept_under_WGS_mask': len(by_ref) / len(w)})
        print(f'  {k:11s} {len(masks[k]):3d}/156 routes   keeps {len(kept):7d}/{len(w):7d} '
              f'post-2018 events ({len(kept)/len(w):5.1%})')
        hist[k] = apply_mask(hist[k], masks[k])
        assert len(hist[k]), f'{k}: mask removed every event'
    pd.DataFrame(audit).to_csv(TABLES / '32_bf_filter_audit.tsv', sep='\t', index=False)

    state_rows, weekly, total_rows = [], [], []
    for k in KEYS:
        a = counts(hist[k], start, end)
        tot = a.sum((1, 2))
        assert np.all(tot > 0), f'{k}: a history has no in-window events'
        total_rows.append({'dataset': k, 'n_histories': a.shape[0], **summary(tot)})
        for i, s in enumerate(STATES):
            for metric, v in (('source_share', a.sum(2)[:, i] / tot),
                              ('sink_share', a.sum(1)[:, i] / tot)):
                state_rows.append({'dataset': k, 'state': s, 'metric': metric, **summary(v)})
        _, w = old.state_and_weekly_summaries(hist[k], k, start,
                                              float(np.nextafter(end, np.inf)))
        weekly.append(w)
        print(f'  {k:11s} {a.shape[0]:4d} histories, {int(tot.sum()):7d} in-window events')
    state = pd.DataFrame(state_rows)
    weekly = pd.concat(weekly)
    for k in KEYS:
        for m in ('source_share', 'sink_share'):
            got = state.query('dataset==@k and metric==@m')['mean'].sum()
            assert np.isclose(got, 1), f'{k}/{m} shares sum to {got}'
    state.to_csv(TABLES / '20_source_sink_post2018_five_groups.tsv', sep='\t', index=False)
    weekly.to_csv(TABLES / '21_weekly_post2018_five_groups.tsv', sep='\t', index=False)
    pd.DataFrame(total_rows).to_csv(TABLES / '22_total_MJ_post2018_five_groups.tsv',
                                    sep='\t', index=False)

    plot_figure(state, weekly, start, end, audit, ref)


def plot_figure(state, weekly, start, end, audit, ref):
    fig = plt.figure(figsize=(19, 28))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 2.25], left=.155, right=.98,
                          bottom=.175, top=.955, hspace=.16, wspace=.62)
    top = gs[0, :].subgridspec(3, 1, hspace=.14)
    axes = [fig.add_subplot(top[i]) for i in range(3)]
    ymax = weekly.weekly_13w_rolling_share_hdi95_high.max() * 105

    for ax, (tag, keys) in zip(axes, PANELS):
        for k in keys:
            q = weekly[weekly.dataset == k]
            x = q.decimal_year.to_numpy()
            ax.fill_between(x, q.weekly_13w_rolling_share_hdi95_low.to_numpy() * 100,
                            q.weekly_13w_rolling_share_hdi95_high.to_numpy() * 100,
                            color=COLOR[k], alpha=.14, lw=0)
            ax.plot(x, q.weekly_13w_rolling_share_mean * 100, color=COLOR[k], lw=1.4,
                    ls='--' if k in ('Calibrated', 'MainCal') else '-')
        ax.set(xlim=(start, end), ylim=(0, ymax), xticks=np.arange(2018, 2027, 1),
               yticks=np.arange(0, ymax, .5))
        ax.text(.012, .90, PANEL_TITLE[tag], transform=ax.transAxes, va='top',
                fontsize=11 * FS, weight='bold', color='#2F3235')
        if len(keys) > 1:
            ax.legend(handles=[Line2D([], [], color=COLOR[k], lw=1.6,
                                      ls='--' if k in ('Calibrated', 'MainCal') else '-',
                                      label='age calibrated' if k in ('Calibrated', 'MainCal')
                                      else 'uncalibrated') for k in keys],
                      loc='upper right', frameon=False, ncol=2, handlelength=2.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        if tag != 'A3':
            ax.tick_params(labelbottom=False)
    axes[0].set_title('A  Temporal dynamic of inferred transitions', loc='left',
                      weight='bold', pad=14)
    temporal_center = (axes[0].get_position().y1 + axes[-1].get_position().y0) / 2
    fig.text(.035, temporal_center, '13-week rolling mean of weekly share (%)', rotation=90,
             ha='center', va='center', fontsize=12 * FS)

    offsets = np.linspace(-.38, .38, len(KEYS))
    for col, metric, title in ((0, 'source_share', 'B  Source composition'),
                               (1, 'sink_share', 'C  Sink composition')):
        ax = fig.add_subplot(gs[1, col])
        q = state[state.metric == metric]
        order = q.groupby('state')['mean'].mean().sort_values(ascending=False).index.tolist()
        for i, s in enumerate(order):
            if i % 2 == 0:
                ax.axhspan(i - .5, i + .5, color='#F4F5F6', zorder=0)
            for off, k in zip(offsets, KEYS):
                r = q[(q.dataset == k) & (q.state == s)].iloc[0]
                y = i + off
                ax.plot([r.lo95 * 100, r.hi95 * 100], [y, y], color=COLOR[k], lw=2.2)
                ax.plot([r.lo50 * 100, r.hi50 * 100], [y, y], color=COLOR[k], lw=10.0,
                        solid_capstyle='butt')
                ax.scatter(r['mean'] * 100, y, color='white', edgecolor=COLOR[k],
                           linewidths=1.8, s=90, zorder=5)
        ax.set_yticks(range(13), [old.STATE_LABEL[s] for s in order])
        ax.invert_yaxis()
        ax.set_xlabel('Share of inferred transitions (%)')
        ax.set_title(title, loc='left', weight='bold')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.grid(axis='x', alpha=.18)

    fig.legend(handles=[Line2D([], [], color=COLOR[k], lw=4.0, marker='o',
                               markerfacecolor='white', markeredgewidth=1.8,
                               markersize=8, label=LABEL[k]) for k in KEYS],
               loc='lower center', bbox_to_anchor=(.55, .105), ncol=3, frameon=False,
               handlelength=2.2, columnspacing=1.6)
    note = (f'Common MJ window: 1 Jan 2018 - {end:.4f}. Mean and 95% HPD; B and C also show '
            '50% HPD. Each share is computed within one posterior history before '
            f'summarization. All five analyses are 13-deme BSSVS. Only routes with Bayes '
            f'factor > {BF_MIN:g} in the same analysis contribute jumps; the threshold is set '
            f'on the WGS reference, where it retains {audit[0]["frac_events_kept"]:.0%} of '
            f'post-2018 events while discarding {156 - len(ref)} of 156 directed routes. '
            'Samples from different analyses are not paired.')
    if SHOW_FOOTNOTE:
        fig.text(.155, .012, textwrap.fill(note, 104), fontsize=9.5 * FS)

    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'Fig2_temporal_source_sink_five_groups_post2018.{ext}',
                    dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('SAVED Fig2_temporal_source_sink_five_groups_post2018')


def replot_saved():
    """Redraw only; leave existing posterior summaries and BF tables unchanged."""
    state = pd.read_csv(TABLES / '20_source_sink_post2018_five_groups.tsv', sep='\t')
    weekly = pd.read_csv(TABLES / '21_weekly_post2018_five_groups.tsv', sep='\t')
    audit = pd.read_csv(TABLES / '32_bf_filter_audit.tsv', sep='\t').set_index('dataset')
    audit = audit.loc[KEYS].reset_index().to_dict('records')
    mask = pd.read_csv(TABLES / '31_route_bf_mask.tsv', sep='\t')
    ref = set(map(tuple, mask.loc[mask.in_WGS_reference_set, ['source', 'sink']].to_numpy()))
    end = min(xml_metadata(XML[k]).date.max() for k in KEYS)
    assert set(state.dataset) == set(weekly.dataset) == set(KEYS)
    assert len(state) == 13 * 2 * len(KEYS)
    plot_figure(state, weekly, 2018.0, end, audit, ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true',
                        help='Use saved summaries without recomputing or rewriting data tables.')
    args = parser.parse_args()
    if args.plot_only:
        replot_saved()
    else:
        main()
