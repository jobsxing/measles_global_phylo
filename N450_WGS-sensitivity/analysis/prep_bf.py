"""Harmonise BSSVS Bayes factors across the five analyses.

Four groups already have a spread.gl `rates` CSV. The root-calibrated 3794 run
never got one, so its BF is recomputed here from the indicator columns of
geo13cal3794_combined.geography.rates.log.

The formula is checked against the four supplied CSVs before it is used: for a
non-reversible K-state BSSVS the prior inclusion probability of any one of the
K(K-1) rates is q = (ln2 + K - 1) / (K(K-1)), and

    BF = [pip / (1 - pip)] / [q / (1 - q)]

Writes tables/30_bf_all_groups.tsv (long, abbreviated deme codes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT
PUB = ROOT
BASE = ROOT
RAW = ROOT
TABLES = ROOT
sys.path.insert(0, str(ROOT))
import analysis_helpers as old

STATES = old.STATE_ORDER[13]
K = 13
Q = (np.log(2) + K - 1) / (K * (K - 1))

CSV = {
    'WGS': ROOT / 'BF_13deme_WGS.csv',
    'N450': ROOT / '13deme.N450.bayes.factor.test.result.csv',
    'Calibrated': ROOT / '13deme_Bayes.factor.test.result.csv',
    'Main': ROOT / 'BSSVS13_region_Bayes.factor.test.result.csv',
}
RATES_LOG = ROOT / 'geo13cal3794_combined.geography.rates.log'
FULLNAME = {v: k for k, v in old.VI_STATE_MAP.items()}


def bf_from_pip(pip):
    pip = np.asarray(pip, float)
    return (pip / (1 - pip)) / (Q / (1 - Q))


def abbrev(s):
    return s.map(lambda x: old.VI_STATE_MAP.get(x, x))


def load_csv(key, path):
    d = pd.read_csv(path)
    d = d.rename(columns={'start_name': 'source', 'end_name': 'sink',
                          'bayes_factor': 'bf', 'posterior_probability': 'pip'})
    d['source'] = abbrev(d['source'])
    d['sink'] = abbrev(d['sink'])
    bad = set(d.source) | set(d.sink)
    assert bad <= set(STATES), f'{key}: unexpected demes {sorted(bad - set(STATES))}'
    assert len(d) == K * (K - 1), f'{key}: {len(d)} rows'
    ref = bf_from_pip(d.pip)
    fin = np.isfinite(ref) & np.isfinite(d.bf) & (d.bf > 0)
    err = np.abs(np.log(ref[fin]) - np.log(d.bf[fin])).max()
    print(f'  {key:11s} max |log BF| discrepancy vs supplied column: {err:.2e}')
    assert err < 1e-3, f'{key}: BF formula does not reproduce the supplied column'
    d['dataset'] = key
    d['bf_is_lower_bound'] = False
    return d[['dataset', 'source', 'sink', 'pip', 'bf', 'bf_is_lower_bound']]


def load_rates_log(key, path):
    """pip = posterior mean of each geography.indicators.<FROM>.<TO> column."""
    d = pd.read_csv(path, sep='\t', comment='#')
    cols = [c for c in d.columns if c.startswith('geography.indicators.')]
    assert len(cols) == K * (K - 1), f'{key}: {len(cols)} indicator columns'
    vals = d[cols].to_numpy(float)
    assert np.isin(vals, (0.0, 1.0)).all(), f'{key}: indicators are not 0/1'
    print(f'  {key:11s} {len(d)} posterior samples, {len(cols)} indicators')
    rows = []
    for c in cols:
        a, b = c[len('geography.indicators.'):].split('.', 1)
        rows.append({'dataset': key, 'source': old.VI_STATE_MAP[a],
                     'sink': old.VI_STATE_MAP[b], 'pip': d[c].mean()})
    out = pd.DataFrame(rows)
    n = len(d)
    out['bf_is_lower_bound'] = out.pip >= 1.0
    out['bf'] = bf_from_pip(np.where(out.pip >= 1.0, n / (n + 1), out.pip))
    print(f'  {key:11s} {int(out.bf_is_lower_bound.sum())} routes at pip = 1 '
          f'(BF reported as a lower bound, >{bf_from_pip(n / (n + 1)):.0f})')
    return out


def main():
    print(f'prior inclusion probability q = {Q:.9f}  (K={K}, non-reversible)')
    print('checking the analytic BF against the four supplied `rates` tables:')
    frames = [load_csv(k, p) for k, p in CSV.items()]
    print('recomputing the missing root-calibrated 3794 table from its rates log:')
    frames.append(load_rates_log('MainCal', RATES_LOG))
    bf = pd.concat(frames, ignore_index=True)
    assert not bf.duplicated(['dataset', 'source', 'sink']).any()
    assert (bf.source != bf.sink).all()

    print('\nsupported routes out of 156, by threshold:')
    print(f'{"dataset":12s}' + ''.join(f'{f"BF>{t}":>9s}' for t in (3, 10, 30, 100, 1000)))
    for k in ['WGS', 'N450', 'Calibrated', 'Main', 'MainCal']:
        q = bf[bf.dataset == k]
        print(f'{k:12s}' + ''.join(f'{int((q.bf > t).sum()):9d}' for t in (3, 10, 30, 100, 1000)))

    wide = bf.pivot_table(index=['source', 'sink'], columns='dataset', values='bf')
    print('\noverlap with the WGS-supported set (BF>3 in WGS):')
    ref = wide[wide['WGS'] > 3]
    print(f'  WGS supports {len(ref)} routes; of those, also BF>3 in:')
    for k in ['N450', 'Calibrated', 'Main', 'MainCal']:
        print(f'    {k:11s} {int((ref[k] > 3).sum()):3d}  ({(ref[k] > 3).mean():.0%})')

    bf.to_csv(TABLES / '30_bf_all_groups.tsv', sep='\t', index=False)
    print(f'\nSAVED {TABLES / "30_bf_all_groups.tsv"}')


if __name__ == '__main__':
    main()
