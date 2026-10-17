#!/usr/bin/env python3
"""Do assets depeg more often in the week after a severe depeg than at other times?

The contagion task (make_contagion.py) ranks which candidates start an episode within a week after a seed
turns severe. This script asks whether that week is special at all. For every seed it moves the decision
hour by +-14, +-21 and +-28 days (placebo hours), applies the same candidate rules at the placebo hour
(make_contagion.candidate_rows: a price in the 24 h before, not collapsed, not inside or within 24 h of
the end of an episode, the seed asset itself excluded) and records whether each candidate starts an
episode within the following week. Placebo rows keep their seed's cluster.

Reported: the follow rate after severe seeds, the placebo rate, their difference and ratio, with 95 %
percentile intervals from a bootstrap over seed clusters (the same draws for both). A second baseline
keeps only calm placebo hours: no severe seed decided within a week before or after the placebo hour; a third
keeps the other placebo hours, those near another severe depeg.

Usage:
  python scripts/contagion_baseline.py                       # main sample -> data/model/contagion_baseline.{json,md}
  python scripts/contagion_baseline.py --labels data/labels_robust/labels_hourly.csv.gz \
      --episodes data/labels_robust/episodes.csv --out data/model_robust/contagion_baseline.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_contagion as mc  # noqa: E402
from common import ROOT, epoch_seconds, known_ts, label_columns, load_registry, not_suspect, observed_only, to_ts  # noqa: E402

H = 3600
DAY = 24 * H
SHIFTS = (-28, -21, -14, 14, 21, 28)
WEEK = mc.FOLLOW_H * H


def rates(df):
    y = df['y168'].to_numpy(dtype=float)
    return float(y.mean()) if len(y) else np.nan


def main():
    ap = argparse.ArgumentParser()
    for name, default in (('registry', 'config/assets.json'), ('rules', 'config/label_rules.json'),
                          ('labels', 'data/labels/labels_hourly.csv.gz'), ('episodes', 'data/labels/episodes.csv')):
        ap.add_argument(f'--{name}', default=str(ROOT / default))
    ap.add_argument('--start', default='2024-01-07')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'model' / 'contagion_baseline.json'))
    args = ap.parse_args()

    rules = json.loads(Path(args.rules).read_text())
    _, assets = load_registry(args.registry)
    sev_level = rules['severity_tiers'][0][0]
    lab = pd.read_csv(args.labels, usecols=label_columns(args.labels))
    lab['ts'] = epoch_seconds(lab['hour'])
    lab['dev_obs'] = observed_only(lab)
    symbols = sorted(lab['symbol'].unique())
    dev = {s: g.set_index('ts')['dev'].sort_index() for s, g in lab.groupby('symbol')}
    dev_obs = {s: g.set_index('ts')['dev_obs'].sort_index() for s, g in lab.groupby('symbol')}
    data_end = int(lab['ts'].max())
    del lab

    ep = not_suspect(pd.read_csv(args.episodes))
    ep = ep[ep['symbol'].isin(symbols)].copy()
    ep['t0'], ep['t1'] = epoch_seconds(ep['start']), epoch_seconds(ep['end'])
    ep['tk'] = np.asarray(known_ts(ep), dtype='int64')
    ep['terminal'] = ep['terminal'].astype(str).str.strip().str.lower().isin(['true', '1', '1.0']) if 'terminal' in ep else False
    seeds = mc.seed_table(ep, dev_obs, 'severe', to_ts(args.start), sev_level)
    va = to_ts(rules['split']['valid_end'][:16])
    first = seeds.groupby('cluster')['tau'].transform('min')
    seeds['split'] = np.where(first <= va, 'train', 'test')

    actual = mc.candidate_rows(seeds, ep, dev, symbols, data_end).dropna(subset=['y168'])
    actual = actual.merge(seeds[['seed_id', 'split']], on='seed_id')
    taus = np.sort(seeds['tau'].to_numpy())
    rows = []
    for d in SHIFTS:
        s2 = seeds.copy()
        s2['tau'] = s2['tau'] + d * DAY
        p = mc.candidate_rows(s2, ep, dev, symbols, data_end).dropna(subset=['y168'])
        p = p.merge(seeds[['seed_id', 'split']], on='seed_id')
        p['shift'] = d
        rows.append(p)
    placebo = pd.concat(rows, ignore_index=True)
    # calm placebo hours: no severe seed decided within a week before or after
    lo = np.searchsorted(taus, placebo['tau'].to_numpy() - WEEK, side='left')
    hi = np.searchsorted(taus, placebo['tau'].to_numpy() + WEEK, side='right')
    placebo['calm'] = (hi - lo) == 0

    def summary(a, p, label):
        out = {'sample': label, 'pairs': int(len(a)), 'followers': int(a['y168'].sum()),
               'follow_rate': rates(a), 'placebo_pairs': int(len(p)), 'placebo_followers': int(p['y168'].sum()),
               'placebo_rate': rates(p)}
        out['diff'] = out['follow_rate'] - out['placebo_rate']
        out['ratio'] = out['follow_rate'] / out['placebo_rate'] if out['placebo_rate'] > 0 else np.nan
        clusters = np.sort(a['cluster'].unique())
        ga = {c: g['y168'].to_numpy(dtype=float) for c, g in a.groupby('cluster')}
        gp = {c: g['y168'].to_numpy(dtype=float) for c, g in p.groupby('cluster')}
        sa = np.array([ga[c].sum() for c in clusters]); na = np.array([len(ga[c]) for c in clusters])
        sp = np.array([gp[c].sum() if c in gp else 0.0 for c in clusters]); npl = np.array([len(gp[c]) if c in gp else 0 for c in clusters])
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, len(clusters), size=(args.boot, len(clusters)))
        fa = sa[draws].sum(1) / na[draws].sum(1)
        fp = sp[draws].sum(1) / np.maximum(npl[draws].sum(1), 1)
        diff, ratio = fa - fp, np.where(fp > 0, fa / np.where(fp > 0, fp, 1), np.nan)
        out['diff_ci'] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        out['ratio_ci'] = [float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))]
        out['share_draws_diff_le_0'] = float((diff <= 0).mean())
        out['clusters'] = int(len(clusters))
        return out

    res = []
    for split in ('all', 'test'):
        a = actual if split == 'all' else actual[actual['split'] == 'test']
        p = placebo if split == 'all' else placebo[placebo['split'] == 'test']
        res.append(summary(a, p, f'{split}: all placebo hours'))
        res.append(summary(a, p[p['calm']], f'{split}: calm placebo hours'))
        res.append(summary(a, p[~p['calm']], f'{split}: placebo hours near another severe depeg'))
        res.append(summary(a, p[p['shift'] < 0], f'{split}: placebo hours before the seed'))
        res.append(summary(a, p[p['shift'] > 0], f'{split}: placebo hours after the seed'))
    by_shift = placebo.groupby('shift')['y168'].agg(['size', 'sum', 'mean']).reset_index()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'results': res, 'by_shift': by_shift.to_dict(orient='records'),
                               'shifts_days': list(SHIFTS), 'boot': args.boot}, indent=1))
    L = ['# Follow rates after severe depegs against placebo hours', '',
         f'Seeds: {seeds["seed_id"].nunique()} severe seeds in {seeds["cluster"].nunique()} clusters; placebo shifts '
         f'{", ".join(f"{d:+d}" for d in SHIFTS)} days; {args.boot} bootstrap draws over seed clusters.', '',
         '| sample | pairs | followers | follow rate | placebo pairs | placebo rate | difference [95 % CI] | ratio [95 % CI] | share of draws with difference <= 0 |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in res:
        L.append(f"| {r['sample']} | {r['pairs']} | {r['followers']} | {r['follow_rate']:.4f} | {r['placebo_pairs']} | "
                 f"{r['placebo_rate']:.4f} | {r['diff']:+.4f} [{r['diff_ci'][0]:+.4f}, {r['diff_ci'][1]:+.4f}] | "
                 f"{r['ratio']:.2f} [{r['ratio_ci'][0]:.2f}, {r['ratio_ci'][1]:.2f}] | {r['share_draws_diff_le_0']:.3f} |")
    L += ['', '| shift (days) | placebo pairs | followers | rate |', '| --- | --- | --- | --- |']
    L += [f"| {int(r['shift']):+d} | {int(r['size'])} | {int(r['sum'])} | {r['mean']:.4f} |" for _, r in by_shift.iterrows()]
    out.with_suffix('.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
