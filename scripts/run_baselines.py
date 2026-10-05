#!/usr/bin/env python3
"""First baselines for the depeg early-warning task (time split, no look-ahead).

Models (trained on the train split, tuned on validation, scored on validation and test):
  base_rate  per-asset share of positive hours in training, smoothed towards the overall share
  rule_dev   the plain deviation rule: score = deepest deviation of the last 6 h (threshold units)
  logit_P    logistic regression on P features (deviation, episode history, market, wrappers, type)
  hgb_P      gradient-boosted trees on P features
  hgb_PE     gradient-boosted trees on P + E (lending exposure) features
  hgb_PEd    P + only the dynamic part of E: changes over 24-168 h, shares and ratios, no USD levels
             (levels grow with Morpho itself and act as an asset fingerprint)
  hgb_E      gradient-boosted trees on E features and asset type only
  hgb_Pid    gradient-boosted trees on P features plus the asset's identity (one indicator per asset):
             a check of the fingerprint account, i.e. whether knowing the asset reproduces what the
             slow-moving exposure levels do
  Tree scores are the mean over --seeds random seeds (default 5; see fit_hgb_seeds).
Metrics, per split:
  ap / roc_auc over all labelled asset-hours; asset_ap = mean AP over assets with positives;
  event recall, alert precision and alert rate at two alert budgets (thresholds = the validation
  scores' top 1 % and 5 %, as a deployed system would set them), and at an equal alert rate of 5 % of
  the split's own asset-hours (the fair comparison of rankings across models); also for severe episodes
  alone (major or collapse, deviation beyond 5 %); median lead time of caught episodes;
  ap_exposed / ap_unexposed on asset-hours with / without lending exposure.

Usage:
  python scripts/run_baselines.py                          # every model, 24 h and 72 h
  python scripts/run_baselines.py --models hgb_PE --horizons 24 --importance
Results are merged into data/model/results.json and rendered to data/model/results.md;
test scores go to data/model/preds_<model>_<h>h.csv.gz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, epoch_seconds, not_suspect, winsor_bounds  # noqa: E402

H = 3600
MODELS = ['base_rate', 'rule_dev', 'logit_P', 'hgb_P', 'hgb_PE', 'hgb_PEd', 'hgb_E', 'hgb_Pid']
DYNAMIC_E = {'mm_blind_oracle_share', 'mm_custom_oracle_share', 'mm_lltv_wavg', 'borrow_to_collateral',
             'family_borrow_to_collateral', 'aave_ltv_max', 'aave_frozen', 'aave_supply_cap_use', 'aave_oracle_gap',
             'has_exposure'}
BUDGETS = (0.01, 0.05)


def need_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit('scikit-learn is needed: pip install scikit-learn')


def feature_sets(groups):
    p, e = groups['P'], groups['E']
    static = [c for c in p if c.startswith('cat_') or c == 'peg_eth']
    dyn = [c for c in e if c.startswith('dlog_') or c in DYNAMIC_E]
    return {'logit_P': p, 'hgb_P': p, 'hgb_PE': p + e, 'hgb_PEd': p + dyn, 'hgb_E': e + static, 'hgb_Pid': list(p)}


# ---------------------------------------------------------------- models
def fit_base_rate(tr, va, te, y, alpha=500.0):
    g = tr.groupby('symbol')[y].agg(['sum', 'size'])
    overall = tr[y].mean()
    rate = (g['sum'] + alpha * overall) / (g['size'] + alpha)
    score = lambda df: df['symbol'].map(rate).fillna(overall).to_numpy(dtype=float)
    return score(va), score(te), {}


def fit_rule(tr, va, te, y):
    score = lambda df: np.nan_to_num(-df['z_min_6h'].to_numpy(dtype=float), nan=0.0)
    return score(va), score(te), {}


def fit_logit(tr, va, te, y, cols):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = tr[cols].astype('float64')
    lo, hi = winsor_bounds(x, 0.001, 0.999)                          # winsorise with training quantiles
    prep = lambda df: df[cols].astype('float64').clip(lo, hi, axis=1)
    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=300))
    model.fit(prep(tr), tr[y].to_numpy())
    return model.predict_proba(prep(va))[:, 1], model.predict_proba(prep(te))[:, 1], {}


def fit_hgb(tr, va, te, y, cols, max_iter, seed=0, every=10):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    cols = [c for c in cols if tr[c].notna().any()]       # newer scikit-learn rejects all-missing features
    model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
                                           random_state=seed)
    model.fit(tr[cols].to_numpy(dtype='float32'), tr[y].to_numpy())
    xv, yv = va[cols].to_numpy(dtype='float32'), va[y].to_numpy()
    best, best_ap, best_sv = 0, -1.0, None
    for i, p in enumerate(model.staged_predict_proba(xv), start=1):   # pick the iteration count on validation
        if i % every == 0 or i == model.n_iter_:
            ap = average_precision_score(yv, p[:, 1])
            if ap > best_ap:
                best, best_ap, best_sv = i, ap, p[:, 1]
    st = None
    for i, p in enumerate(model.staged_predict_proba(te[cols].to_numpy(dtype='float32')), start=1):
        if i == best:
            st = p[:, 1]
            break
    return best_sv, st, {'best_iter': best, 'model': model, 'cols': cols}


def fit_hgb_seeds(tr, va, te, y, cols, max_iter, seeds):
    """fit_hgb averaged over random seeds. On the onset table the seed matters: scikit-learn places the
    trees' bin thresholds on a random draw of 200,000 training rows, and that draw alone moves a single
    fit's test AP by several thousandths. Each seed picks its own iteration count on validation; the
    validation and test scores are the mean over seeds."""
    svs, sts, iters, first = [], [], [], None
    for s in seeds:
        sv, st, info = fit_hgb(tr, va, te, y, cols, max_iter, seed=s)
        svs.append(sv)
        sts.append(st)
        iters.append(info['best_iter'])
        first = first or info
    return (np.mean(svs, axis=0), np.mean(sts, axis=0),
            {'best_iter': iters, 'model': first['model'], 'cols': first['cols']})


# ---------------------------------------------------------------- metrics
def split_episodes(episodes, df):
    lo, hi = df['ts'].min(), df['ts'].max()
    return episodes[(episodes['ts'] > lo) & (episodes['ts'] <= hi + 72 * H)]


def evaluate(df, score, y, h, episodes, thresholds):
    from sklearn.metrics import average_precision_score, roc_auc_score
    yy = df[y].to_numpy()
    out = {'rows': int(len(df)), 'positives': int(yy.sum())}
    if 0 < yy.sum() < len(yy):
        out['ap'] = float(average_precision_score(yy, score))
        out['roc_auc'] = float(roc_auc_score(yy, score))
    out['prevalence'] = float(yy.mean())
    per = []
    for sym, idx in df.groupby('symbol').indices.items():
        ys = yy[idx]
        if 0 < ys.sum() < len(ys):
            per.append(average_precision_score(ys, score[idx]))
    out['asset_ap'] = float(np.mean(per)) if per else None
    out['assets_with_positives'] = len(per)
    for name, mask in (('exposed', df['has_exposure'].to_numpy() > 0), ('unexposed', df['has_exposure'].to_numpy() == 0)):
        ys = yy[mask]
        out[f'ap_{name}'] = float(average_precision_score(ys, score[mask])) if 10 <= ys.sum() < len(ys) else None
        out[f'prevalence_{name}'] = float(ys.mean()) if len(ys) else None
    # event level: an episode is caught if some hour in [start - h, start - 1 h] raises an alert
    eps = split_episodes(episodes, df)
    ts_all = df['ts'].to_numpy()
    by_sym = df.groupby('symbol').indices
    top = np.zeros(len(score), bool)               # exactly the top 5 % of this split's asset-hours (ties broken by time)
    top[np.argsort(-score, kind='stable')[:int(round(0.05 * len(score)))]] = True
    alerts = [(f'budget_{b:g}', score >= thr, float(thr)) for b, thr in thresholds.items()] + [('rate_0.05', top, None)]
    severity = eps['severity'] if 'severity' in eps else pd.Series('', index=eps.index)
    for key, alert, thr in alerts:
        res = {'threshold': thr, 'alert_rate': float(alert.mean()),
               'alert_precision': float(yy[alert].mean()) if alert.any() else None}
        caught, leads, n, sev_n, sev_caught = 0, [], 0, 0, 0
        for sym, s, sev in zip(eps['symbol'], eps['ts'], severity):
            if sym not in by_sym:
                continue
            idx = by_sym[sym]
            win = idx[(ts_all[idx] >= s - h * H) & (ts_all[idx] <= s - H)]
            if not len(win):
                continue
            n += 1
            severe = sev in ('major', 'collapse')
            sev_n += severe
            hit = win[alert[win]]
            if len(hit):
                caught += 1
                sev_caught += severe
                leads.append((s - ts_all[hit].min()) / H)
        res.update({'episodes': n, 'event_recall': caught / n if n else None,
                    'severe_episodes': sev_n, 'severe_recall': sev_caught / sev_n if sev_n else None,
                    'median_lead_h': float(np.median(leads)) if leads else None})
        out[key] = res
    return out


def importance(model, va, y, cols, n=60000, seed=0):
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(va[y].to_numpy() == 1)
    neg = np.flatnonzero(va[y].to_numpy() == 0)
    idx = np.concatenate([pos, rng.choice(neg, size=min(len(neg), max(n - len(pos), 0)), replace=False)])
    x, yy = va[cols].to_numpy(dtype='float32')[idx], va[y].to_numpy()[idx]
    r = permutation_importance(model, x, yy, scoring='average_precision', n_repeats=3, random_state=seed)
    order = np.argsort(-r.importances_mean)
    return [{'feature': cols[i], 'ap_drop': float(r.importances_mean[i]), 'std': float(r.importances_std[i])} for i in order[:20]]


# ---------------------------------------------------------------- report
def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '–'
    return f'{100 * v:.1f}%' if pct else f'{v:.3f}'


def render(results, path):
    lines = ['# Baseline results', '',
             'Train to 2025-03, validation 2025-04 to 2025-09, test 2025-10 to 2026-09. AP = average precision '
             '(the share of positive hours is the AP of a random score). Recall = share of episodes with an alert in '
             'the h hours before their start. "5 % alert rate": every model alerts on its top 5 % of test asset-hours '
             '(fair comparison); "validation threshold": the top 5 % of validation scores, as deployed, with the '
             'test alert rate it produced in brackets.', '']
    for h in sorted({r['horizon'] for r in results}):
        rs = sorted([r for r in results if r['horizon'] == h], key=lambda r: MODELS.index(r['model']) if r['model'] in MODELS else 99)
        t0 = rs[0]['test']
        lines += [f'## {h} h horizon (test: {t0["rows"]:,} asset-hours, {t0["positives"]:,} positive, '
                  f'prevalence {100 * t0["prevalence"]:.2f}%)', '',
                  '| model | valid AP | test AP | test ROC-AUC | asset-avg AP | recall (5 % alert rate) | severe recall | '
                  'alert precision | median lead (h) | recall @ validation threshold (test alert rate) | AP exposed | AP unexposed |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in rs:
            t, b5, eq = r['test'], r['test']['budget_0.05'], r['test'].get('rate_0.05', {})
            lead = eq.get('median_lead_h')
            lines.append(f"| {r['model']} | {fmt(r['valid'].get('ap'))} | {fmt(t.get('ap'))} | {fmt(t.get('roc_auc'))} | "
                         f"{fmt(t.get('asset_ap'))} | {fmt(eq.get('event_recall'), True)} | {fmt(eq.get('severe_recall'), True)} | "
                         f"{fmt(eq.get('alert_precision'), True)} | {'–' if lead is None else f'{lead:.0f}'} | "
                         f"{fmt(b5['event_recall'], True)} ({fmt(b5['alert_rate'], True)}) | "
                         f"{fmt(t.get('ap_exposed'))} | {fmt(t.get('ap_unexposed'))} |")
        lines += ['', f"Test episodes: {t0['budget_0.05']['episodes']} ({t0['budget_0.05'].get('severe_episodes')} severe); exposed asset-hours have prevalence "
                      f"{fmt(t0.get('prevalence_exposed'), True)}, unexposed {fmt(t0.get('prevalence_unexposed'), True)}.", '']
        for e in rs:
            if e.get('importance'):
                lines += [f"Permutation importance of {e['model']} on validation (drop in AP, top 20):", '',
                          '| feature | AP drop |', '| --- | --- |']
                lines += [f"| {x['feature']} | {x['ap_drop']:.4f} |" for x in e['importance']]
                lines.append('')
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'model' / 'dataset.pkl.gz'))
    ap.add_argument('--episodes', default=str(ROOT / 'data' / 'labels' / 'episodes.csv'))
    ap.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    ap.add_argument('--horizons', nargs='+', type=int, default=[24, 72])
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--seeds', type=int, default=5, help='tree scores are averaged over this many random seeds')
    ap.add_argument('--importance', action='store_true', help='permutation importance for hgb_PE and hgb_PEd')
    ap.add_argument('--out', default=str(ROOT / 'data' / 'model'))
    args = ap.parse_args()
    need_sklearn()

    d = pd.read_pickle(args.data)
    groups = json.loads((Path(args.data).parent / 'dataset_features.json').read_text())
    sets = feature_sets(groups)
    ep = not_suspect(pd.read_csv(args.episodes))
    ep['ts'] = epoch_seconds(ep['start'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / 'results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else []

    for h in args.horizons:
        y = f'y{h}'
        m = d[d[y].notna()]
        tr, va, te = (m[m['split'] == s].reset_index(drop=True) for s in ('train', 'valid', 'test'))
        id_cols = []
        if 'hgb_Pid' in args.models:
            ids = sorted(m['symbol'].unique())
            id_cols = [f'id_{s}' for s in ids]
            tr, va, te = (pd.concat([f, pd.DataFrame((f['symbol'].to_numpy()[:, None] == np.array(ids)[None, :]).astype('float32'),
                                                     columns=id_cols)], axis=1) for f in (tr, va, te))
        for name in args.models:
            t0 = time.time()
            info = {}
            if name == 'base_rate':
                sv, st, info = fit_base_rate(tr, va, te, y)
            elif name == 'rule_dev':
                sv, st, info = fit_rule(tr, va, te, y)
            elif name.startswith('logit'):
                sv, st, info = fit_logit(tr, va, te, y, sets[name])
            else:
                cols = sets[name] + (id_cols if name == 'hgb_Pid' else [])
                sv, st, info = fit_hgb_seeds(tr, va, te, y, cols, args.max_iter, range(args.seeds))
            thresholds = {b: np.quantile(sv, 1 - b) for b in BUDGETS}
            entry = {'model': name, 'horizon': h, 'n_features': len(sets.get(name, [])) + (len(id_cols) if name == 'hgb_Pid' else 0), 'seconds': None,
                     'best_iter': info.get('best_iter'),
                     'valid': evaluate(va, sv, y, h, ep, thresholds), 'test': evaluate(te, st, y, h, ep, thresholds)}
            if args.importance and name in ('hgb_PE', 'hgb_PEd'):
                entry['importance'] = importance(info['model'], va, y, info['cols'])
            entry['seconds'] = round(time.time() - t0, 1)
            pd.DataFrame({'symbol': te['symbol'], 'ts': te['ts'], 'y': te[y].astype(int), 'has_exposure': te['has_exposure'],
                          'score': st}).to_csv(out / f'preds_{name}_{h}h.csv.gz', index=False)
            results = [r for r in results if not (r['model'] == name and r['horizon'] == h)] + [entry]
            res_path.write_text(json.dumps(results, indent=1))
            t = entry['test']
            print(f"{name:10s} {h:2d}h  valid AP {entry['valid'].get('ap', float('nan')):.3f}  test AP {t.get('ap', float('nan')):.3f}  "
                  f"ROC-AUC {t.get('roc_auc', float('nan')):.3f}  recall@5% {fmt(t['budget_0.05']['event_recall'], True)}  "
                  f"({entry['seconds']} s)")
    render(results, out / 'results.md')
    print(f'wrote {out / "results.md"}')


if __name__ == '__main__':
    main()
