#!/usr/bin/env python3
"""Economic magnitude of the warnings: lending exposure at stake and how much of it the alerts precede.

For each test-period onset episode (the episodes the onset metrics count: a test asset-hour in the 24 h before
the start), the collateral that the asset's family had posted on Morpho, Aave and Spark one hour before the
start (family_collateral_usd in data/graph_hourly/token_features.csv.gz, as the onset features see it) is the
lending exposure at stake. With alerts on the top 5 % of test asset-hours (as in Table 4) the script reports:
  - event recall weighted by that collateral, for the price trees, logistic regression and the deviation rule,
    against the expected recall of random alerts on as many asset-hours;
  - the false-alarm burden: alert runs (stretches of consecutive alerted hours on one asset) per day and the
    share of runs that contain an hour followed by a start within 24 h, against random alerts (50 draws).
For escalation, the collateral at the decision hour on the test-period depegs that reach -5 % and on those
the depth flag (the deepest fifth of new depegs) catches. Depegs count as having collateral when at least
--min-collateral (default $10,000) was posted; weighted recall uses every dollar.

Usage:
  python scripts/economic_magnitude.py              # writes data/model/economic_magnitude.{json,md}
  python scripts/economic_magnitude.py --model-dir data/model_robust --episodes data/labels_robust/episodes.csv
  python scripts/economic_magnitude.py --model-dir data/model_traded --episodes data/labels_traded/episodes.csv
"""
import argparse
import json
import sys
from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, epoch_seconds, not_suspect  # noqa: E402

H = 3600
MODELS = {'hgb_P': 'trees, price', 'logit_P': 'logistic regression, price', 'rule_dev': 'deviation rule'}


def top_alerts(p, budget=0.05):
    """Exactly the top round(budget * N) scores, ties broken by row order (as run_baselines does)."""
    top = np.zeros(len(p), bool)
    top[np.argsort(-p['score'].to_numpy(), kind='stable')[:int(round(budget * len(p)))]] = True
    return top


def p_random(w, k, n):
    """Probability that k random alerts out of n asset-hours hit at least one of w given hours."""
    if w <= 0:
        return 0.0
    if n - w < k:
        return 1.0
    return 1.0 - np.exp(lgamma(n - w + 1) - lgamma(n - w - k + 1) + lgamma(n - k + 1) - lgamma(n + 1))


def runs(sym, ts, y):
    """Alert runs: consecutive alerted hours of one asset. Returns (number of runs, runs with a positive hour)."""
    d = pd.DataFrame({'s': sym, 't': ts, 'y': y}).sort_values(['s', 't'])
    new = (d['s'] != d['s'].shift()) | (d['t'].diff() != H)
    rid = new.cumsum()
    g = d.groupby(rid)['y'].max()
    return int(len(g)), int(g.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=str(ROOT / 'data' / 'model'))
    ap.add_argument('--episodes', default=str(ROOT / 'data' / 'labels' / 'episodes.csv'))
    ap.add_argument('--features', default=str(ROOT / 'data' / 'graph_hourly' / 'token_features.csv.gz'))
    ap.add_argument('--draws', type=int, default=50)
    ap.add_argument('--min-collateral', type=float, default=1e4, help='USD posted for a depeg to count as having collateral')
    args = ap.parse_args()
    md = Path(args.model_dir)
    tf = pd.read_csv(args.features, usecols=['symbol', 't', 'family_collateral_usd', 'mm_borrow_against_usd'])
    tf = tf.set_index(['symbol', 't'])
    ep = not_suspect(pd.read_csv(args.episodes))
    ep['ts'] = epoch_seconds(ep['start'])
    out = {}
    L = ['# Economic magnitude of the warnings (test period)', '']
    for h in (24, 72):
        base = pd.read_csv(md / f'preds_hgb_P_{h}h.csv.gz', float_precision='round_trip')
        lo, hi = base['ts'].min(), base['ts'].max()
        e = ep[(ep['ts'] > lo) & (ep['ts'] <= hi + 72 * H)].copy()
        by_sym = {s: g['ts'].to_numpy() for s, g in base.groupby('symbol')}
        e['w'] = [int(((by_sym.get(s, np.array([])) >= t - h * H) & (by_sym.get(s, np.array([])) <= t - H)).sum())
                  for s, t in zip(e['symbol'], e['ts'])]
        e = e[e['w'] > 0].reset_index(drop=True)                       # the episodes the onset metrics count
        key = pd.MultiIndex.from_arrays([e['symbol'], e['ts'] - H])
        e['collateral'] = tf['family_collateral_usd'].reindex(key).fillna(0).to_numpy()
        e['mm_loans'] = tf['mm_borrow_against_usd'].reindex(key).fillna(0).to_numpy()
        e['severe'] = e['severity'].isin(['major', 'collapse'])
        n, k = len(base), int(round(0.05 * len(base)))
        e['p_random'] = [p_random(w, k, n) for w in e['w']]
        cw = e['collateral'].to_numpy()
        ex = cw >= args.min_collateral
        res = {'episodes': int(len(e)), 'min_collateral_usd': args.min_collateral, 'with_collateral': int(ex.sum()),
               'with_any_collateral': int((cw > 0).sum()), 'collateral_total_usd': float(cw.sum()),
               'top1_share': float(np.max(cw) / cw.sum()) if cw.sum() else None,
               'top5_share': float(np.sort(cw)[::-1][:5].sum() / cw.sum()) if cw.sum() else None,
               'collateral_median_usd_exposed': float(np.median(cw[ex])) if ex.any() else None,
               'severe': int(e['severe'].sum()), 'severe_with_collateral': int((ex & e['severe']).sum()),
               'severe_collateral_usd': float(cw[e['severe'].to_numpy()].sum()),
               'random': {'recall': float(e['p_random'].mean()), 'recall_with_collateral': float(e.loc[ex, 'p_random'].mean()),
                          'recall_without_collateral': float(e.loc[~ex, 'p_random'].mean()),
                          'collateral_weighted_recall': float((e['p_random'] * cw).sum() / cw.sum()) if cw.sum() else None},
               'top_episodes': e.sort_values('collateral', ascending=False).head(8)[
                   ['symbol', 'start', 'severity', 'collateral', 'mm_loans']].to_dict(orient='records'),
               'models': {}}
        days = pd.to_datetime(base['ts'], unit='s', utc=True).dt.floor('D').nunique()
        for m in MODELS:
            p = pd.read_csv(md / f'preds_{m}_{h}h.csv.gz', float_precision='round_trip')
            assert len(p) == n
            al = top_alerts(p)
            alerted = {s: g['ts'].to_numpy() for s, g in p[al].groupby('symbol')}
            caught = np.array([bool(((alerted.get(s, np.array([])) >= t - h * H) & (alerted.get(s, np.array([])) <= t - H)).any())
                               for s, t in zip(e['symbol'], e['ts'])])
            nr, nr_pos = runs(p.loc[al, 'symbol'].to_numpy(), p.loc[al, 'ts'].to_numpy(), p.loc[al, 'y'].to_numpy())
            res['models'][m] = {
                'recall': float(caught.mean()), 'caught': int(caught.sum()),
                'recall_with_collateral': float(caught[ex].mean()), 'recall_without_collateral': float(caught[~ex].mean()),
                'collateral_weighted_recall': float(cw[caught].sum() / cw.sum()) if cw.sum() else None,
                'collateral_caught_usd': float(cw[caught].sum()),
                'exposed_episodes_caught': int((caught & ex).sum()),
                'severe_collateral_weighted_recall': float(cw[caught & e['severe'].to_numpy()].sum() / cw[e['severe'].to_numpy()].sum())
                if cw[e['severe'].to_numpy()].sum() else None,
                'alert_precision': float(p.loc[al, 'y'].mean()), 'runs': nr, 'runs_per_day': nr / days,
                'runs_with_start': nr_pos, 'run_precision': nr_pos / nr if nr else None}
        rng = np.random.default_rng(0)
        rr = []
        for _ in range(args.draws):
            idx = rng.choice(n, size=k, replace=False)
            nr, nr_pos = runs(base['symbol'].to_numpy()[idx], base['ts'].to_numpy()[idx], base['y'].to_numpy()[idx])
            rr.append((nr, nr_pos))
        rr = np.array(rr, float)
        res['random'].update({'runs_per_day': float(rr[:, 0].mean() / days), 'run_precision': float((rr[:, 1] / rr[:, 0]).mean()),
                              'alert_precision': float(base['y'].mean())})
        res['days'] = int(days)
        out[f'onset_{h}h'] = res
        L += [f'## Onset, {h} h horizon: {len(e)} test episodes, {res["with_collateral"]} with at least '
              f'${args.min_collateral:,.0f} of collateral posted one hour before the start ({res["with_any_collateral"]} with any; '
              f'total ${cw.sum() / 1e6:,.1f}M, median ${(res["collateral_median_usd_exposed"] or 0) / 1e6:,.2f}M over the '
              f'{res["with_collateral"]})', '',
              f'Severe episodes: {res["severe"]}, {res["severe_with_collateral"]} with collateral (${res["severe_collateral_usd"] / 1e6:,.1f}M).', '',
              f'The largest holds {res["top1_share"]:.1%} and the five largest {res["top5_share"]:.1%} of the collateral summed '
              'over episodes.', '',
              '| alerts (top 5 % of test asset-hours) | recall | recall, with collateral | recall, without | collateral-weighted recall | '
              'severe, collateral-weighted | alert precision | runs a day | runs with a start within the horizon |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for m, r in res['models'].items():
            L.append(f"| {MODELS[m]} | {r['recall']:.1%} | {r['recall_with_collateral']:.1%} | {r['recall_without_collateral']:.1%} | "
                     f"{r['collateral_weighted_recall']:.1%} | "
                     f"{(r['severe_collateral_weighted_recall'] or float('nan')):.1%} | {r['alert_precision']:.1%} | "
                     f"{r['runs_per_day']:.1f} | {r['run_precision']:.1%} ({r['runs_with_start']} of {r['runs']}) |")
        rd = res['random']
        L += [f"| random (expected) | {rd['recall']:.1%} | {rd['recall_with_collateral']:.1%} | {rd['recall_without_collateral']:.1%} | "
              f"{rd['collateral_weighted_recall']:.1%} | – | {rd['alert_precision']:.1%} | "
              f"{rd['runs_per_day']:.1f} | {rd['run_precision']:.1%} |", '', 'Largest exposures at a start:', '',
              '| asset | start | severity | collateral | Morpho loans against it |', '| --- | --- | --- | --- | --- |']
        L += [f"| {t['symbol']} | {t['start'][:16]} | {t['severity']} | ${t['collateral'] / 1e6:,.1f}M | ${t['mm_loans'] / 1e6:,.1f}M |"
              for t in res['top_episodes']]
        L.append('')

    # escalation: the depth flag
    et = pd.read_csv(md / 'escalation.csv.gz')
    l1 = et[(et['landmark'] == 1) & et['y_severe'].notna()]
    te = l1[l1['split'] == 'test'].copy()
    k = int(round(0.2 * len(te)))
    flagged = set(te.sort_values('depth_frac', ascending=False, kind='stable').head(k)['episode_id'])
    key = pd.MultiIndex.from_arrays([te['symbol'], te['ts']])
    te['collateral'] = tf['family_collateral_usd'].reindex(key).fillna(0).to_numpy()
    te['flag'] = te['episode_id'].isin(flagged)
    esc = te[te['y_severe'] == 1]
    r = {'test_episodes': int(len(te)), 'escalate': int(len(esc)), 'flagged': k, 'flagged_escalate': int(esc['flag'].sum()),
         'escalate_with_collateral': int((esc['collateral'] >= args.min_collateral).sum()), 'escalate_collateral_usd': float(esc['collateral'].sum()),
         'caught_collateral_usd': float(esc.loc[esc['flag'], 'collateral'].sum()),
         'flagged_collateral_usd': float(te.loc[te['flag'], 'collateral'].sum()),
         'all_collateral_usd': float(te['collateral'].sum()),
         'escalations': esc.sort_values('collateral', ascending=False)[['symbol', 'start', 'collateral', 'flag']].to_dict(orient='records')}
    r['caught_share'] = r['caught_collateral_usd'] / r['escalate_collateral_usd'] if r['escalate_collateral_usd'] else None
    out['escalation_1h'] = r
    L += ['## Escalation, decided one hour in (test period)', '',
          f"{r['escalate']} of {r['test_episodes']} test depegs reach -5 %; {r['escalate_with_collateral']} of them had at least ${args.min_collateral:,.0f} of collateral posted "
          f"at the decision hour, ${r['escalate_collateral_usd'] / 1e6:,.1f}M in all. The depth flag (deepest {k}) catches "
          f"{r['flagged_escalate']} escalations, holding ${r['caught_collateral_usd'] / 1e6:,.1f}M "
          f"({(r['caught_share'] or 0):.1%} of the collateral on escalating depegs). Collateral on all flagged depegs: "
          f"${r['flagged_collateral_usd'] / 1e6:,.1f}M of ${r['all_collateral_usd'] / 1e6:,.1f}M on all test depegs.", '',
          '| asset | start | collateral | flagged |', '| --- | --- | --- | --- |']
    L += [f"| {x['symbol']} | {x['start'][:16]} | ${x['collateral'] / 1e6:,.2f}M | {'yes' if x['flag'] else 'no'} |" for x in r['escalations']]
    (md / 'economic_magnitude.json').write_text(json.dumps(out, indent=1, default=float))
    (md / 'economic_magnitude.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
