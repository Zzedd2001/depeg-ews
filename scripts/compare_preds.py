#!/usr/bin/env python3
"""Paired comparison of two onset models on the test split, with an asset-block bootstrap.

Reads data/model/preds_<model>_<h>h.csv.gz (run_baselines.py) and reports AP of each model and the AP
difference with a 95 % interval, over all test asset-hours and separately over hours with and without
lending exposure. Assets are resampled with replacement (all hours of an asset together), the same
draws for both models; AP is computed with each asset's draw count as row weight, so one sort serves
every draw.

Usage:
  python scripts/compare_preds.py --horizons 24 72 --pairs hgb_PE:hgb_P hgb_PEd:hgb_P
Writes data/model/compare_preds.md (and .json); a run with other horizons or pairs keeps the results it did not redo.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402


class SortedAP:
    """AP of one score vector under arbitrary non-negative row weights (tie-aware, as in scikit-learn)."""

    def __init__(self, y, score):
        self.order = np.argsort(-score, kind='mergesort')
        s = score[self.order]
        self.y = y[self.order].astype(float)
        self.last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]      # last row of each run of equal scores

    def __call__(self, w):
        w = w[self.order]
        tp = np.cumsum(w * self.y)[self.last]
        n = np.cumsum(w)[self.last]
        if tp[-1] <= 0:
            return np.nan
        keep = n > 0
        precision = np.where(keep, tp / np.where(keep, n, 1), 0.0)
        return float(np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * precision))


def compare(a, b, B, seed):
    """a, b: frames with symbol, ts, y, has_exposure, score on the same rows."""
    m = a.merge(b[['symbol', 'ts', 'score']], on=['symbol', 'ts'], suffixes=('_a', '_b'), validate='one_to_one')
    codes, assets = pd.factorize(m['symbol'])
    y = m['y'].to_numpy(dtype=float)
    out = {}
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(assets), len(assets)), minlength=len(assets)) for _ in range(B)])
    for name, mask in (('all', np.ones(len(m), bool)), ('exposed', m['has_exposure'].to_numpy() > 0),
                       ('unexposed', m['has_exposure'].to_numpy() == 0)):
        if y[mask].sum() < 10:
            continue
        fa, fb = SortedAP(y[mask], m.loc[mask, 'score_a'].to_numpy()), SortedAP(y[mask], m.loc[mask, 'score_b'].to_numpy())
        ones = np.ones(int(mask.sum()))
        ap_a, ap_b = fa(ones), fb(ones)
        cm = codes[mask]
        diffs = []
        for c in counts:
            w = c[cm].astype(float)
            da, db = fa(w), fb(w)
            if not (np.isnan(da) or np.isnan(db)):
                diffs.append(da - db)
        diffs = np.asarray(diffs)
        out[name] = {'rows': int(mask.sum()), 'positives': int(y[mask].sum()), 'assets': int(len(np.unique(cm))),
                     'ap_a': ap_a, 'ap_b': ap_b, 'diff': ap_a - ap_b,
                     'ci': [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
                     'share_above_0': float(np.mean(diffs > 0)), 'draws': int(len(diffs))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(ROOT / 'data' / 'model'))
    ap.add_argument('--horizons', nargs='+', type=int, default=[24, 72])
    ap.add_argument('--pairs', nargs='+', default=['hgb_PE:hgb_P', 'hgb_PEd:hgb_P', 'hgb_Pid:hgb_P', 'hgb_PE:hgb_Pid',
                                                    'logit_P:hgb_P', 'hgb_P:rule_dev'])
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    d = Path(args.dir)
    old = json.loads((d / 'compare_preds.json').read_text()) if (d / 'compare_preds.json').exists() else []
    res = []
    for h in args.horizons:
        for pair in args.pairs:
            a_name, b_name = pair.split(':')
            fa, fb = d / f'preds_{a_name}_{h}h.csv.gz', d / f'preds_{b_name}_{h}h.csv.gz'
            if not (fa.exists() and fb.exists()):
                print(f'skip {pair} {h}h: predictions missing')
                continue
            r = compare(pd.read_csv(fa), pd.read_csv(fb), args.bootstrap, args.seed)
            res.append({'horizon': h, 'a': a_name, 'b': b_name, 'subsets': r})
            for k, v in r.items():
                print(f"{h:2d}h {a_name} - {b_name} [{k}]: {v['ap_a']:.3f} - {v['ap_b']:.3f} = {v['diff']:+.3f} "
                      f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    done = {(r['horizon'], r['a'], r['b']) for r in res}
    res = sorted([r for r in old if (r['horizon'], r['a'], r['b']) not in done] + res, key=lambda r: r['horizon'])
    (d / 'compare_preds.json').write_text(json.dumps(res, indent=1))
    lines = ['# Paired model comparisons (onset, test split)', '',
             f'AP difference A - B with 95 % intervals from an asset-block bootstrap ({args.bootstrap} draws, '
             'the same draws for both models). "exposed": asset-hours with lending exposure.', '',
             '| horizon | A - B | subset | asset-hours (positive) | AP A | AP B | difference [95 % CI] | share of draws > 0 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in res:
        for k, v in r['subsets'].items():
            lines.append(f"| {r['horizon']} h | {r['a']} - {r['b']} | {k} | {v['rows']:,} ({v['positives']:,}) | {v['ap_a']:.3f} | "
                         f"{v['ap_b']:.3f} | {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {v['share_above_0']:.2f} |")
    (d / 'compare_preds.md').write_text('\n'.join(lines) + '\n')
    print(f"wrote {d / 'compare_preds.md'}")


if __name__ == '__main__':
    main()
