#!/usr/bin/env python3
"""Semi-synthetic power check for the contagion graph links, on the real seed-candidate pairs.

Everything real is kept (seeds, candidates, features, the time split, seed clusters and the link
structure); only the labels are simulated. Each pair follows with probability p = its own-state
probability (a logistic regression on the OWN block fitted to the real labels), raised for pairs with
any graph link (outside the seed's family) by a relative risk RR, capped at 0.95. For each RR and
replicate the script reruns the time-split comparison of run_contagion.py (logit_graph against
logit_fam, per-seed AP over seeds with at least one follower, 95 % interval from resampling seed
clusters) and records two detections:
  ranking   the interval of the AP difference lies above zero
  rates     the follow rate of linked pairs exceeds that of unlinked pairs, interval above zero
RR = 1 gives the false-detection rate. With links nonzero for few seeds, power tells how strong a
lending channel would have had to be for the paper's tests to see it.

Usage:
  python scripts/power_contagion.py --rr 1 2 4 8 --reps 100           # about 1-2 minutes per RR
Writes data/model/contagion_power.json and contagion_power.md (results for several runs are merged by RR).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_contagion as rc  # noqa: E402
from common import ROOT  # noqa: E402

MODELS = ['logit_fam', 'logit_graph']


def own_probability(d, y, groups):
    """In-sample follow probability from the candidate's own state (the null model of no lending channel)."""
    return rc.fit_logit(d, d, y, groups['OWN'])


def interval_above_zero(v, clusters, B, rng):
    """Seed-cluster bootstrap of the mean of v (one value per seed); True when the 95 % interval is above 0."""
    uc = np.unique(clusters)
    pos = pd.Series(np.arange(len(uc)), index=uc)[clusters].to_numpy()
    draws = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))[pos].astype(float)
        if w.sum() > 0:
            draws.append((w @ v) / w.sum())
    lo = float(np.quantile(draws, 0.025))
    return lo > 0, lo


def one_replicate(d, groups, p, rng, B):
    d = d.assign(ysim=(rng.uniform(size=len(d)) < p).astype(float))
    mask, preds = rc.predict(d, 'ysim', MODELS, groups, 'time')
    frame = d[mask].reset_index(drop=True)
    pr = {m: v[mask] for m, v in preds.items()}
    ps = rc.per_seed(frame, 'ysim', pr, MODELS)
    out = {'followers': int(d['ysim'].sum()), 'linked_followers': int(d.loc[d['linked'], 'ysim'].sum()),
           'test_seeds': int(len(ps))}
    if ps.empty:
        return {**out, 'dap': np.nan, 'ranking': False, 'rates': False}
    v = (ps['logit_graph:ap'] - ps['logit_fam:ap']).to_numpy()
    out['dap'] = float(v.mean())
    out['ranking'], out['dap_lo'] = interval_above_zero(v, ps['cluster'].to_numpy(), B, rng)
    lr = rc.link_rates(d, 'ysim', d['linked'].to_numpy(), B=B, seed=int(rng.integers(1 << 31)))
    out['rate_diff'] = lr.get('difference', np.nan)
    out['rates'] = bool(lr.get('difference_ci', [0, 0])[0] > 0)
    return out


def run(args):
    data = Path(args.data)
    d = pd.read_csv(data)
    groups = json.loads((data.parent / 'contagion_features.json').read_text())
    y = f'y{args.window}'
    d = d[d[y].notna()].reset_index(drop=True)
    d['linked'] = d[groups['GRAPH']].gt(0).any(axis=1) & (d['same_family'] == 0)
    p0 = own_probability(d, y, groups)
    observed = rc.link_rates(d, y, d['linked'].to_numpy(), B=10)['difference']     # the real linked - unlinked follow rate
    out_dir = Path(args.out)
    res_path = out_dir / 'contagion_power.json'
    results = {float(r['rr']): r for r in (json.loads(res_path.read_text()) if res_path.exists() else [])}
    for rr in args.rr:
        t0 = time.time()
        p = np.where(d['linked'], np.minimum(p0 * rr, 0.95), p0)
        rng = np.random.default_rng(args.seed + int(round(rr * 100)))
        reps = [one_replicate(d, groups, p, rng, args.bootstrap) for _ in range(args.reps)]
        r = pd.DataFrame(reps)
        results[float(rr)] = {
            'rr': float(rr), 'reps': int(len(r)), 'window': args.window, 'pairs': int(len(d)),
            'linked_pairs': int(d['linked'].sum()), 'linked_test_pairs': int((d['linked'] & (d['split'] != 'train')).sum()),
            'mean_followers': float(r['followers'].mean()), 'mean_linked_followers': float(r['linked_followers'].mean()),
            'mean_dap': float(r['dap'].mean()), 'power_ranking': float(r['ranking'].mean()),
            'mean_rate_diff': float(r['rate_diff'].mean()), 'power_rates': float(r['rates'].mean()),
            'observed_rate_diff': float(observed), 'share_at_or_below_observed': float((r['rate_diff'] <= observed).mean()),
            'seconds': round(time.time() - t0, 1)}
        print(f"RR {rr:>4}: linked followers {r['linked_followers'].mean():.1f}, mean dAP {r['dap'].mean():+.4f}, "
              f"power ranking {r['ranking'].mean():.2f}, rates {r['rates'].mean():.2f}  ({time.time() - t0:.0f} s)")
    rows = sorted(results.values(), key=lambda r: r['rr'])
    res_path.write_text(json.dumps(rows, indent=1))
    lines = ['# Semi-synthetic power of the contagion link tests', '',
             f"Real pairs ({rows[0]['pairs']:,}), features, time split and links ({rows[0]['linked_pairs']} linked pairs, "
             f"{rows[0]['linked_test_pairs']} in the test period); simulated followers within {args.window} h: own-state "
             'probability, times RR for linked pairs (capped at 0.95). Detection: ranking = 95 % seed-cluster interval of '
             'the per-seed AP difference logit_graph - logit_fam above zero (time split); rates = interval of the follow-rate '
             'difference between linked and unlinked pairs above zero. RR = 1 is the false-detection rate. Last column: share of '
             f"replicates whose rate difference is at or below the observed one ({rows[0]['observed_rate_diff']:+.3f}), "
             'i.e. how compatible the real data are with a channel of that strength.', '',
             '| RR | replicates | followers per replicate | linked followers | mean ΔAP | power (ranking) | mean rate difference | power (rates) | at or below observed |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['rr']:g} | {r['reps']} | {r['mean_followers']:.0f} | {r['mean_linked_followers']:.1f} | "
                     f"{r['mean_dap']:+.4f} | {r['power_ranking']:.2f} | {r['mean_rate_diff']:+.3f} | {r['power_rates']:.2f} | "
                     f"{r.get('share_at_or_below_observed', float('nan')):.2f} |")
    (out_dir / 'contagion_power.md').write_text('\n'.join(lines) + '\n')
    print(f"wrote {out_dir / 'contagion_power.md'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'model' / 'contagion_severe.csv.gz'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'model'))
    ap.add_argument('--window', type=int, default=168)
    ap.add_argument('--rr', nargs='+', type=float, default=[1, 2, 4, 8])
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--bootstrap', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
