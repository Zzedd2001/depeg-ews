#!/usr/bin/env python3
"""How much does hourly monitoring add over a daily early-warning model?

A daily monitor sees one price per asset and day. This script builds such a monitor at the daily decision
points of the onset table (00:00 UTC, the asset-hours of data/model/dataset.pkl.gz whose hour is 00:00)
and compares it with the hourly models' scores at the same asset-days. Labels are the onset labels y24 / y72
(an episode starts within the next 24 / 72 h). Splits are those of the hourly table.

Daily features (from daily closes only, the 00:00 deviation in units of the label threshold):
  close now; min, max, mean and std of closes over 3 to 30 days; change over 1 and 7 days; days with a close
  below -0.5 (near misses) or -1 (a close beyond the threshold) over 30 to 365 days and days since the last;
  share of missing closes; market context from closes (share of assets below -0.5, mean USD deviation, closes
  beyond the threshold across assets over 1 and 7 days), ETH returns over 1 and 7 days and 30-day volatility of
  daily ETH returns; the underlying's close for wrappers; peg and category.

Daily models:
  d_rule   the close's deviation (score = -close)
  d_tail   a daily tail-risk model in the spirit of Lee (2026): AR(1) of the daily deviation with an
           exponentially weighted (RiskMetrics, lambda 0.94) variance of its residuals, and a generalized Pareto
           lower tail of the standardized residuals beyond their 10 % quantile (McNeil and Frey, 2000); the score
           is the probability that the next close is beyond the threshold. AR parameters are fitted per asset on
           training days (pooled over assets of the same peg when an asset has fewer than 90 training days).
  d_logit  logistic regression on the daily features (same pipeline as logit_P)
  d_hgb    gradient-boosted trees on the daily features; the leaf size (20, 50, 100 or 200 days) and the
           iteration count are chosen on validation AP, and scores are the mean over 5 random seeds
  d_logit_H, d_hgb_H  the same with the hourly table's episode-history features (episodes known from hourly
           prices) instead of the close-based history: what remains of the hourly gain once history is shared

Hourly models compared (their test scores at the 00:00 asset-hours, preds_<model>_<h>h.csv.gz): logit_P,
hgb_P, rule_dev. Paired AP differences use the asset-block bootstrap of compare_preds.py.

Usage:
  python scripts/daily_baseline.py            # writes data/model/daily_baseline.{json,md}, preds_daily_*.csv.gz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_preds as cp  # noqa: E402
from common import ROOT, epoch_seconds, load_registry, winsor_bounds  # noqa: E402

H, DAY = 3600, 86400
HIST_H = ['episodes_30d', 'episodes_90d', 'episodes_365d', 'hours_since_episode', 'mkt_episodes_24h', 'mkt_episodes_168h']


def daily_features(z, eth, under, meta, symbols):
    """z: days x assets frame of close deviations (threshold units), complete daily index; eth: daily ETH close."""
    f = {'dz_now': z}
    for w in (3, 7, 30):
        f[f'dz_min_{w}d'] = z.rolling(w, min_periods=1).min()
    f['dz_max_7d'] = z.rolling(7, min_periods=1).max()
    for w in (7, 30):
        f[f'dz_mean_{w}d'] = z.rolling(w, min_periods=1).mean()
        f[f'dz_std_{w}d'] = z.rolling(w, min_periods=3).std()
    for k in (1, 7):
        f[f'dz_chg_{k}d'] = z - z.shift(k)
    f['dz_vs_mean_30d'] = z - f['dz_mean_30d']
    near = (z < -0.5).astype(float).where(z.notna())
    beyond = (z <= -1).astype(float).where(z.notna())
    f['d_near_30d'] = near.rolling(30, min_periods=1).sum()
    for w in (30, 90, 365):
        f[f'd_beyond_{w}d'] = beyond.rolling(w, min_periods=1).sum()
    for name, m in (('d_days_since_near', near), ('d_days_since_beyond', beyond)):
        idx = np.arange(len(m), dtype=float)[:, None]
        last = pd.DataFrame(np.where(m.fillna(0).to_numpy() > 0, idx, np.nan), index=m.index, columns=m.columns).ffill()
        f[name] = (idx - last).clip(upper=365)
    f['d_missing_frac_7d'] = z.isna().astype(float).rolling(7, min_periods=1).mean()
    usd = [s for s in symbols if meta[s]['peg'] == 'USD']
    mk = {'d_mkt_stress_frac': (z < -0.5).sum(axis=1) / z.notna().sum(axis=1).replace(0, np.nan),
          'd_mkt_mean_z_usd': z[usd].clip(-5, 5).mean(axis=1),
          'd_mkt_beyond_1d': beyond.sum(axis=1), 'd_mkt_beyond_7d': beyond.sum(axis=1).rolling(7, min_periods=1).sum()}
    le = np.log(eth)
    mk['d_eth_ret_1d'] = le - le.shift(1)
    mk['d_eth_ret_7d'] = le - le.shift(7)
    mk['d_eth_vol_30d'] = (le - le.shift(1)).rolling(30, min_periods=10).std()
    for name, s in mk.items():
        f[name] = pd.DataFrame(np.repeat(s.to_numpy()[:, None], len(symbols), axis=1), index=z.index, columns=symbols)
    for name, src in (('d_und_now', z), ('d_und_min_7d', f['dz_min_7d'])):
        f[name] = pd.DataFrame({s: (src[under[s]] if s in under else np.nan) for s in symbols}, index=z.index)
    return f


def fit_tail(z, peg_of, train_end, lam=0.94, tail_q=0.10, min_days=90):
    """AR(1) + EWMA variance + GPD lower tail; returns a days x assets frame of P(next close <= -1) and fit info."""
    from scipy.stats import genpareto
    zz = z.ffill(limit=3)
    tr = zz.index <= train_end
    def ols(x, y):
        X = np.c_[np.ones_like(x), x]
        a, b = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(a), float(np.clip(b, 0.0, 0.999))
    pairs = {}
    for s in zz.columns:
        x, y = zz[s].shift(1).to_numpy(), zz[s].to_numpy()
        ok = ~np.isnan(x) & ~np.isnan(y) & tr
        pairs[s] = (x[ok], y[ok])
    pooled = {}
    for peg in sorted(set(peg_of.values())):
        xs = np.concatenate([pairs[s][0] for s in zz.columns if peg_of[s] == peg])
        ys = np.concatenate([pairs[s][1] for s in zz.columns if peg_of[s] == peg])
        pooled[peg] = ols(xs, ys)
    ab = {s: (ols(*pairs[s]) if len(pairs[s][0]) >= min_days else pooled[peg_of[s]]) for s in zz.columns}
    # EWMA variance of one-step residuals, run causally over the whole sample
    mu = pd.DataFrame({s: ab[s][0] + ab[s][1] * zz[s] for s in zz.columns}, index=zz.index)  # forecast of next close
    resid = zz - mu.shift(1)
    sig2 = pd.DataFrame(np.nan, index=zz.index, columns=zz.columns)
    init = float(np.nanmedian(resid[tr].to_numpy() ** 2))   # robust start; the recursion forgets it within weeks
    for s in zz.columns:
        r = resid[s].to_numpy()
        v, out = init, np.full(len(r), np.nan)
        for i in range(len(r)):
            if not np.isnan(r[i]):
                v = lam * v + (1 - lam) * r[i] ** 2
            out[i] = v                         # variance forecast for the next day, known at day i
        sig2[s] = out
    sd = np.sqrt(sig2)
    u = (resid / sd.shift(1))[tr].to_numpy().ravel()
    u = u[~np.isnan(u)]
    q = np.quantile(u, tail_q)
    exc = q - u[u < q]
    c, loc, scale = genpareto.fit(exc, floc=0.0)
    srt = np.sort(u)

    def cdf(v):
        v = np.asarray(v, float)
        emp = np.searchsorted(srt, v, side='right') / len(srt)
        tail = tail_q * genpareto.sf(q - v, c, loc=0.0, scale=scale)
        return np.where(v < q, tail, emp)

    std_threshold = (-1.0 - mu) / sd
    p = pd.DataFrame(cdf(std_threshold.to_numpy()), index=zz.index, columns=zz.columns).where(std_threshold.notna())
    info = {'lambda': lam, 'tail_quantile': tail_q, 'gpd_shape': float(c), 'gpd_scale': float(scale), 'u_threshold': float(q),
            'assets_own_ar': int(sum(len(pairs[s][0]) >= min_days for s in zz.columns)), 'pooled_ar': pooled}
    return p, info


def fit_logit(tr, va, te, y, cols):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = tr[cols].astype('float64')
    lo, hi = winsor_bounds(x, 0.001, 0.999)
    prep = lambda df: df[cols].astype('float64').clip(lo, hi, axis=1)
    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=500))
    model.fit(prep(tr), tr[y].to_numpy())
    return model.predict_proba(prep(va))[:, 1], model.predict_proba(prep(te))[:, 1], {}


def fit_hgb(tr, va, te, y, cols, seeds=range(5), leaves=(20, 50, 100, 200), max_iter=300, every=10):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    cols = [c for c in cols if tr[c].notna().any()]
    xt, yt = tr[cols].to_numpy(dtype='float32'), tr[y].to_numpy()
    xv, yv = va[cols].to_numpy(dtype='float32'), va[y].to_numpy()
    xe = te[cols].to_numpy(dtype='float32')

    def one(leaf, seed):
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=31, min_samples_leaf=leaf,
                                           l2_regularization=1.0, early_stopping=False, random_state=seed).fit(xt, yt)
        best, best_ap, best_sv = 0, -1.0, None
        for i, p in enumerate(m.staged_predict_proba(xv), start=1):
            if i % every == 0 or i == m.n_iter_:
                a = average_precision_score(yv, p[:, 1])
                if a > best_ap:
                    best, best_ap, best_sv = i, a, p[:, 1]
        st = next(p[:, 1] for i, p in enumerate(m.staged_predict_proba(xe), start=1) if i == best)
        return best_ap, best_sv, st, best

    # leaf size chosen on validation AP of the seed-averaged scores
    cand = {}
    for leaf in leaves:
        runs = [one(leaf, s) for s in seeds]
        sv, st = np.mean([r[1] for r in runs], axis=0), np.mean([r[2] for r in runs], axis=0)
        cand[leaf] = (average_precision_score(yv, sv), sv, st, [r[3] for r in runs])
    leaf = max(cand, key=lambda k: cand[k][0])
    return cand[leaf][1], cand[leaf][2], {'leaf': leaf, 'iters': cand[leaf][3],
                                          'valid_ap_by_leaf': {str(k): float(v[0]) for k, v in cand.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=str(ROOT / 'config' / 'assets.json'))
    ap.add_argument('--wrappers', default=str(ROOT / 'config' / 'wrappers.json'))
    ap.add_argument('--rules', default=str(ROOT / 'config' / 'label_rules.json'))
    ap.add_argument('--labels', default=str(ROOT / 'data' / 'labels' / 'labels_hourly.csv.gz'))
    ap.add_argument('--prices', default=str(ROOT / 'data' / 'prices_llama_hourly.csv.gz'))
    ap.add_argument('--data', default=str(ROOT / 'data' / 'model' / 'dataset.pkl.gz'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'model'))
    ap.add_argument('--bootstrap', type=int, default=1000)
    args = ap.parse_args()
    from sklearn.metrics import average_precision_score, roc_auc_score

    rules = json.loads(Path(args.rules).read_text())
    _, assets = load_registry(args.registry)
    meta = {a['symbol']: a for a in assets}
    lab = pd.read_csv(args.labels, usecols=['hour', 'symbol', 'dev'])
    lab['ts'] = epoch_seconds(lab['hour'])
    symbols = sorted(lab['symbol'].unique())
    thr = pd.Series({s: rules['threshold'][meta[s]['peg']] for s in symbols})
    day0 = -(-lab['ts'].min() // DAY) * DAY            # first 00:00 UTC in the data
    days = pd.Index(np.arange(day0, lab['ts'].max() + 1, DAY, dtype='int64'), name='ts')
    dev = lab[lab['ts'] % DAY == 0].pivot(index='ts', columns='symbol', values='dev').reindex(index=days, columns=symbols)
    z = dev.clip(-1, 1) / thr
    del lab
    px = pd.read_csv(args.prices, usecols=['symbol', 'ts', 'price'])
    weth = px[px['symbol'] == 'WETH'].sort_values('ts')
    del px
    eth = pd.merge_asof(pd.DataFrame({'ts': days.to_numpy()}), weth[['ts', 'price']].astype({'ts': 'int64'}),
                        on='ts', direction='backward', tolerance=2 * H).set_index('ts')['price']
    wr = json.loads(Path(args.wrappers).read_text()) if Path(args.wrappers).exists() else {'pairs': []}
    under = {w: u for w, u in wr.get('pairs', []) if w in symbols and u in symbols}
    feats = daily_features(z, eth, under, meta, symbols)
    train_end = int(pd.Timestamp(rules['split']['train_end'][:16], tz='UTC').timestamp())
    ptail, tail_info = fit_tail(z, {s: meta[s]['peg'] for s in symbols}, train_end)
    feats['d_tail'] = ptail

    d = pd.read_pickle(args.data)
    rows = d[d['ts'] % DAY == 0].reset_index(drop=True)
    del d
    ri, ci = days.get_indexer(rows['ts']), pd.Index(symbols).get_indexer(rows['symbol'])
    assert (ri >= 0).all() and (ci >= 0).all()
    cols = {name: fr.to_numpy(dtype='float64')[ri, ci].astype('float32') for name, fr in feats.items()}
    rows = pd.concat([rows, pd.DataFrame(cols, index=rows.index)], axis=1)
    static = ['peg_eth'] + [c for c in rows.columns if c.startswith('cat_')]
    dcols = [c for c in feats if c != 'd_tail'] + static
    hist_daily = ['d_beyond_30d', 'd_beyond_90d', 'd_beyond_365d', 'd_days_since_beyond', 'd_mkt_beyond_1d', 'd_mkt_beyond_7d']
    dcols_h = [c for c in dcols if c not in hist_daily] + HIST_H

    out = Path(args.out)
    results, preds = {'rows': {}, 'models': {}, 'tail_model': tail_info, 'features_daily': dcols, 'features_daily_H': dcols_h}, {}
    for h in (24, 72):
        y = f'y{h}'
        m = rows[rows[y].notna()]
        tr, va, te = (m[m['split'] == s].reset_index(drop=True) for s in ('train', 'valid', 'test'))
        results['rows'][h] = {s: {'rows': int(len(f)), 'positives': int(f[y].sum())} for s, f in (('train', tr), ('valid', va), ('test', te))}
        fits = {'d_rule': (-va['dz_now'].to_numpy(), -te['dz_now'].to_numpy(), {}),
                'd_tail': (va['d_tail'].fillna(0).to_numpy(), te['d_tail'].fillna(0).to_numpy(), {}),
                'd_logit': fit_logit(tr, va, te, y, dcols),
                'd_hgb': fit_hgb(tr, va, te, y, dcols),
                'd_logit_H': fit_logit(tr, va, te, y, dcols_h),
                'd_hgb_H': fit_hgb(tr, va, te, y, dcols_h)}
        for name, (sv, st, info) in fits.items():
            yv, yt = va[y].to_numpy(), te[y].to_numpy()
            r = {'valid_ap': float(average_precision_score(yv, sv)), 'test_ap': float(average_precision_score(yt, st)),
                 'test_roc_auc': float(roc_auc_score(yt, st)), 'info': {k: v for k, v in info.items()}}
            results['models'][f'{name}_{h}h'] = r
            f = pd.DataFrame({'symbol': te['symbol'], 'ts': te['ts'], 'y': te[y].astype(int), 'has_exposure': te['has_exposure'], 'score': st})
            f.to_csv(out / f'preds_daily_{name}_{h}h.csv.gz', index=False)
            preds[(name, h)] = f
            print(f'{name:10s} {h}h valid AP {r["valid_ap"]:.3f} test AP {r["test_ap"]:.3f} ROC-AUC {r["test_roc_auc"]:.3f} {info.get("leaf", "")}')
        for name in ('logit_P', 'hgb_P', 'rule_dev'):
            hp = pd.read_csv(out / f'preds_{name}_{h}h.csv.gz')
            hp = hp[hp['ts'] % DAY == 0].reset_index(drop=True)
            assert len(hp) == len(te), (name, len(hp), len(te))
            preds[(name, h)] = hp
            results['models'][f'{name}_{h}h_at_00'] = {'test_ap': float(average_precision_score(hp['y'], hp['score'])),
                                                       'test_roc_auc': float(roc_auc_score(hp['y'], hp['score']))}
            print(f'{name:10s} {h}h at 00:00 test AP {results["models"][f"{name}_{h}h_at_00"]["test_ap"]:.3f}')
    pairs = [('logit_P', 'd_logit'), ('hgb_P', 'd_hgb'), ('logit_P', 'd_tail'), ('hgb_P', 'd_tail'), ('rule_dev', 'd_rule'),
             ('logit_P', 'd_logit_H'), ('hgb_P', 'd_hgb_H'), ('d_logit', 'd_tail'), ('d_hgb', 'd_tail')]
    comp = []
    for h in (24, 72):
        for a, b in pairs:
            r = cp.compare(preds[(a, h)], preds[(b, h)], args.bootstrap, 0)['all']
            comp.append({'horizon': h, 'a': a, 'b': b, **r})
            print(f"{h}h {a} - {b}: {r['ap_a']:.3f} - {r['ap_b']:.3f} = {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
    results['comparisons'] = comp
    (out / 'daily_baseline.json').write_text(json.dumps(results, indent=1, default=float))
    L = ['# Daily versus hourly monitoring (onset, test split, decision hour 00:00 UTC)', '',
         'Daily models see one close per asset and day; hourly models are scored at the same asset-days. '
         f'AP differences A - B with 95 % intervals from an asset-block bootstrap ({args.bootstrap} draws).', '']
    for h in (24, 72):
        rr = results['rows'][h]
        L += [f'## {h} h horizon (test {rr["test"]["rows"]:,} asset-days, {rr["test"]["positives"]} positive)', '',
              '| model | valid AP | test AP | test ROC-AUC |', '| --- | --- | --- | --- |']
        for k, v in results['models'].items():
            if k.endswith(f'_{h}h') or k.endswith(f'_{h}h_at_00'):
                L.append(f"| {k} | {v.get('valid_ap', float('nan')):.3f} | {v['test_ap']:.3f} | {v['test_roc_auc']:.3f} |")
        L += ['', '| A - B | AP A | AP B | difference [95 % CI] | share of draws > 0 |', '| --- | --- | --- | --- | --- |']
        L += [f"| {c['a']} - {c['b']} | {c['ap_a']:.3f} | {c['ap_b']:.3f} | {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | "
              f"{c['share_above_0']:.2f} |" for c in comp if c['horizon'] == h]
        L.append('')
    (out / 'daily_baseline.md').write_text('\n'.join(L) + '\n')
    print(f"wrote {out / 'daily_baseline.md'}")


if __name__ == '__main__':
    main()
