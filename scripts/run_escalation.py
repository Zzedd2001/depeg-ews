#!/usr/bin/env python3
"""Escalation task: a depeg has just started and is not severe yet. Will it reach -5 % (held for two hours),
or last at least a day? Reads data/model/escalation.csv.gz (make_escalation.py).

Models (fixed in advance, nothing tuned on test data):
  prior        a constant: the reference a random ranking reaches (AP = share of positives, ROC-AUC = 0.5)
  depth        depth so far, as a fraction of the 5 % severity level
  asset_hist   the asset's own share of earlier severe (or long) episodes, Beta(1, 4)-smoothed
  logit_core   logistic regression on 13 episode, market and history features (CORE)
  logit_coreE  the same plus 6 lending-exposure features of the asset family (CORE_E)
  hgb_P        gradient-boosted trees on all episode (EP) and P features
  hgb_PE       the same plus all E features
Protocols:
  time   fit on episodes that started by 2025-09-30, score those that started later (as deployed)
  loao   leave one asset out over all years: each asset is scored by models that never saw it, so
         exposure features cannot help by recognising the asset
Metrics: AP, ROC-AUC, Brier score, 95 % intervals from an asset-cluster bootstrap, and paired differences
(logit_coreE - logit_core, hgb_PE - hgb_P, logit_core - depth) on the same resamples.
Inference: logistic regression of the outcome on 6 controls (CONTROLS, few enough for the number of severe
episodes) and the exposure block CORE_E, standardised, with standard errors clustered by asset: odds ratio
per standard deviation and a joint Wald test of the exposure block, with p-values from a pairs cluster
bootstrap-t as well (the chi2 reference over-rejects with a few dozen clusters). Three samples: all
episodes; episodes of assets with lending exposure; all episodes with asset-category dummies (is it
exposure or asset type?). Coefficients beyond 5 log-odds per SD are marked as (quasi-)separated.

Usage:
  python scripts/run_escalation.py                                  # every landmark in the table, both outcomes
  python scripts/run_escalation.py --landmarks 1 --outcomes severe --bootstrap 2000
Writes data/model/escalation_results.json (merged per landmark x outcome), escalation_results.md and
escalation_preds_<outcome>_<L>h.csv.gz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, winsor_bounds  # noqa: E402

MODELS = ['prior', 'depth', 'asset_hist', 'logit_core', 'logit_coreE', 'hgb_P', 'hgb_PE']
PROBABILISTIC = {'prior', 'asset_hist', 'logit_core', 'logit_coreE', 'hgb_P', 'hgb_PE'}
CORE = ['depth_frac', 'ep_z_first', 'ep_hours_below', 'ep_back_in_band', 'pre_z_min_24h', 'peg_eth', 'und_below',
        'mkt_stress_frac', 'eth_ret_24h', 'log_prior_episodes', 'prior_severe_share', 'prior_long_share',
        'log_asset_age_days']
CORE_E = ['has_exposure', 'log_family_collateral_usd', 'family_borrow_to_collateral', 'mm_blind_oracle_share',
          'log_family_vault_exposure_usd', 'dlog_family_borrow_against_usd_24h']
CONTROLS = ['depth_frac', 'ep_z_first', 'peg_eth', 'mkt_stress_frac', 'prior_severe_share', 'log_asset_age_days']
CATEGORIES = ['cat_synthetic', 'cat_rwa_backed', 'cat_yield_bearing', 'cat_cdp', 'cat_lst', 'cat_lrt']
PAIRS = [('logit_coreE', 'logit_core'), ('hgb_PE', 'hgb_P'), ('logit_core', 'depth')]
OUTCOMES = {'severe': 'y_severe', 'long': 'y_long'}
HIST = {'severe': 'prior_severe_share', 'long': 'prior_long_share'}
PROTOCOLS = ('time', 'loao')


# ---------------------------------------------------------------- metrics (tie-aware, as in scikit-learn)
def ap_score(y, s):
    order = np.argsort(-s, kind='mergesort')
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]          # last row of each run of equal scores
    tp = np.cumsum(y)[last]
    if tp[-1] == 0:
        return np.nan
    precision, recall = tp / (last + 1), tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def auc_score(y, s):
    from scipy.stats import rankdata
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def point_metrics(y, s, probabilistic):
    return {'ap': ap_score(y, s), 'roc_auc': auc_score(y, s),
            'brier': float(np.mean((s - y) ** 2)) if probabilistic else None}


def cluster_bootstrap(groups, y, preds, B, seed=0):
    """Resample assets with replacement; metric draws per model, aligned across models."""
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    draws = {m: {'ap': [], 'roc_auc': []} for m in preds}
    for _ in range(B):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        for m, s in preds.items():
            draws[m]['ap'].append(ap_score(yb, s[idx]))
            draws[m]['roc_auc'].append(auc_score(yb, s[idx]))
    return {m: {k: np.asarray(v) for k, v in d.items()} for m, d in draws.items()}


def ci(v):
    v = v[~np.isnan(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [None, None]


# ---------------------------------------------------------------- models
def core_matrix(tr, te, with_e):
    """Core features: winsorised at the training 1st / 99th percentiles, missing -> training median.
    Exposure features: missing means no exposure -> 0, then winsorised the same way."""
    def prep(cols, fill_median):
        a, b = tr[cols].astype(float), te[cols].astype(float)
        if not fill_median:
            a, b = a.fillna(0.0), b.fillna(0.0)
        (lo, hi), med = winsor_bounds(a, 0.01, 0.99), a.median()
        return (a.clip(lo, hi, axis=1).fillna(med).fillna(0.0), b.clip(lo, hi, axis=1).fillna(med).fillna(0.0))
    xtr, xte = prep(CORE, True)
    if with_e:
        etr, ete = prep(CORE_E, False)
        xtr, xte = pd.concat([xtr, etr], axis=1), pd.concat([xte, ete], axis=1)
    return xtr.to_numpy(), xte.to_numpy()


def fit_logit(tr, te, y, with_e, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    xtr, xte = core_matrix(tr, te, with_e)
    sc = StandardScaler().fit(xtr)
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(xtr), tr[y].to_numpy())
    return m.predict_proba(sc.transform(xte))[:, 1]


def fit_hgb(tr, te, y, cols, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cols = [c for c in cols if tr[c].notna().any()]       # newer scikit-learn rejects all-missing features
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, max_depth=3, min_samples_leaf=20,
                                       l2_regularization=1.0, early_stopping=False, random_state=seed)
    m.fit(tr[cols].to_numpy(dtype='float32'), tr[y].to_numpy())
    return m.predict_proba(te[cols].to_numpy(dtype='float32'))[:, 1]


def score(name, tr, te, y, outcome, sets, constant):
    if name == 'prior':
        return np.full(len(te), constant)
    if name in ('logit_core', 'logit_coreE', 'hgb_P', 'hgb_PE') and tr[y].nunique() < 2:
        return np.full(len(te), tr[y].mean())             # a training fold with one class: nothing to learn
    if name == 'depth':
        return te['depth_frac'].fillna(0.0).to_numpy(dtype=float)
    if name == 'asset_hist':
        return te[HIST[outcome]].to_numpy(dtype=float)
    if name in ('logit_core', 'logit_coreE'):
        return fit_logit(tr, te, y, name == 'logit_coreE')
    return fit_hgb(tr, te, y, sets[name])


def run_protocol(d, y, outcome, models, sets, protocol):
    """Out-of-sample scores for the rows a protocol evaluates; returns (mask of scored rows, {model: scores})."""
    preds = {m: np.full(len(d), np.nan) for m in models}
    if protocol == 'time':
        fit = d['split'].isin(['train', 'valid']).to_numpy()
        test = ~fit
        for m in models:
            preds[m][test] = score(m, d[fit], d[test], y, outcome, sets, d.loc[fit, y].mean())
        return test, preds
    for sym in sorted(d['symbol'].unique()):              # leave one asset out
        test = (d['symbol'] == sym).to_numpy()
        for m in models:                                  # 'prior' gets one constant for every fold
            preds[m][test] = score(m, d[~test], d[test], y, outcome, sets, d[y].mean())
    return np.ones(len(d), bool), preds


# ---------------------------------------------------------------- inference
def logit_clustered(X, y, groups, ridge=1e-4, iters=100):
    """Logistic regression by damped Newton steps (step halving keeps the penalised log-likelihood
    rising, so a separated feature grows slowly instead of throwing the others off; a tiny ridge on the
    slopes keeps it finite), with clustered sandwich standard errors (small-sample factor
    G/(G-1) * (N-1)/(N-K), as in Stata)."""
    n, k = X.shape
    X1 = np.column_stack([np.ones(n), X])
    pen = np.r_[0.0, np.full(k, ridge)]
    beta = np.zeros(k + 1)
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    loglik = lambda b: float(np.sum(y * (X1 @ b) - np.logaddexp(0.0, X1 @ b)) - 0.5 * np.sum(pen * b * b))
    ll = loglik(beta)
    converged = False
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ beta, -35, 35)))
        hess = (X1 * (p * (1 - p))[:, None]).T @ X1 + np.diag(pen)
        step = np.linalg.solve(hess, X1.T @ (y - p) - pen * beta)
        t = 1.0
        while True:
            new = beta + t * step
            new_ll = loglik(new)
            if new_ll >= ll - 1e-12 or t < 1e-6:
                break
            t /= 2
        beta, gain, ll = new, new_ll - ll, new_ll
        if np.max(np.abs(t * step)) < 1e-9 or abs(gain) < 1e-12:
            converged = True
            break
    p = 1 / (1 + np.exp(-np.clip(X1 @ beta, -35, 35)))
    hess = (X1 * (p * (1 - p))[:, None]).T @ X1 + np.diag(pen)
    inv = np.linalg.inv(hess)
    codes = np.unique(groups, return_inverse=True)[1]
    G = int(codes.max()) + 1
    u = X1 * (y - p)[:, None]
    score_g = np.column_stack([np.bincount(codes, weights=u[:, j], minlength=G) for j in range(k + 1)])
    V = inv @ (score_g.T @ score_g) @ inv * (G / (G - 1)) * ((n - 1) / (n - k - 1))
    return beta, V, converged, G


def bootstrap_t(X, y, groups, beta, V, eidx, B, seed=0):
    """Pairs cluster bootstrap-t (Cameron, Gelbach and Miller 2008): resample assets, refit, and compare
    |beta* - beta| / se* with the observed |t|; for the exposure block the same with the Wald statistic.
    More reliable than the normal / chi2 reference with a few dozen clusters."""
    p_t, p_w, done = bootstrap_t_sets(X, y, groups, beta, V, {'block': eidx} if eidx else {}, B, seed)
    return p_t, p_w.get('block'), done


def bootstrap_t_sets(X, y, groups, beta, V, sets, B, seed=0):
    """bootstrap_t with a joint Wald test for each named set of coefficient indices ({name: [j, ...]}),
    all on the same resamples. Returns (p per coefficient, {name: p}, draws used)."""
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    se = np.sqrt(np.diag(V))
    t_obs = np.abs(beta / se)
    sets = {k: list(v) for k, v in sets.items() if len(v)}
    w_obs = {k: float(beta[e] @ np.linalg.solve(V[np.ix_(e, e)], beta[e])) for k, e in sets.items()}
    hit_t, hit_w, done = np.zeros(len(beta)), dict.fromkeys(sets, 0), 0
    for _ in range(B):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        yb = y[idx]
        if yb.sum() < 2 or len(yb) - yb.sum() < 2:
            continue
        gb = np.repeat(np.arange(len(pick)), [len(blocks[i]) for i in pick])
        bb, Vb, _, _ = logit_clustered(X[idx], yb, gb)
        seb = np.sqrt(np.clip(np.diag(Vb), 1e-300, None))
        hit_t += np.abs((bb - beta) / seb) >= t_obs
        for k, e in sets.items():
            dlt = bb[e] - beta[e]
            try:
                hit_w[k] += float(dlt @ np.linalg.solve(Vb[np.ix_(e, e)], dlt)) >= w_obs[k]
            except np.linalg.LinAlgError:
                hit_w[k] += 1
        done += 1
    return (hit_t + 1) / (done + 1), {k: (h + 1) / (done + 1) for k, h in hit_w.items()}, done


def inference(d, y, outcome, spec='all', boot=999):
    """spec 'all': controls + exposure block over all episodes; 'exposed': only assets with lending exposure;
    'category': controls + asset-category dummies (largest category as reference, peg_eth then redundant)."""
    from scipy.stats import chi2, norm
    rows = d[d[y].notna()]
    controls = [HIST[outcome] if c == 'prior_severe_share' else c for c in CONTROLS]
    ecols = list(CORE_E)
    if spec == 'exposed':
        rows = rows[rows['has_exposure'] > 0]
        ecols.remove('has_exposure')
    if spec == 'category':
        cats = [c for c in CATEGORIES if c in rows and rows[c].std() > 0]
        if cats:
            ref = max(cats, key=lambda c: rows[c].sum())
            controls = [c for c in controls if c != 'peg_eth'] + [c for c in cats if c != ref]
    ctrl = rows[controls].astype(float)
    ctrl = ctrl.clip(*winsor_bounds(ctrl, 0.01, 0.99), axis=1).fillna(ctrl.median())
    e = rows[ecols].astype(float).fillna(0.0)
    e = e.clip(*winsor_bounds(e, 0.01, 0.99), axis=1)
    X = pd.concat([ctrl, e], axis=1)
    X = X.loc[:, X.std() > 1e-9]                       # columns that are constant in this sample
    names = list(X.columns)
    yy = rows[y].to_numpy(dtype=float)
    out = {'spec': spec, 'n': int(len(rows)), 'positives': int(yy.sum()), 'assets': int(rows['symbol'].nunique()),
           'dropped': [c for c in controls + ecols if c not in names]}
    if yy.sum() < 5 or len(rows) - yy.sum() < 5:
        return out
    Xs = ((X - X.mean()) / X.std()).to_numpy()
    groups = rows['symbol'].to_numpy()
    beta, V, conv, G = logit_clustered(Xs, yy, groups)
    se = np.sqrt(np.diag(V))
    eidx = [j for j, c in enumerate(names, start=1) if c in CORE_E]
    p_boot, pw_boot, done = bootstrap_t(Xs, yy, groups, beta, V, eidx, boot) if boot else (None, None, 0)
    out.update({'clusters': G, 'converged': conv, 'events_per_variable': float(min(yy.sum(), len(yy) - yy.sum()) / len(names)),
                'bootstrap_draws': done, 'coef': []})
    for j, c in enumerate(names, start=1):
        z = beta[j] / se[j]
        out['coef'].append({'feature': c, 'block': 'E' if c in CORE_E else 'control', 'beta_sd': float(beta[j]),
                            'se': float(se[j]), 'odds_ratio': float(np.exp(np.clip(beta[j], -50, 50))),
                            'or_ci': [float(np.exp(np.clip(beta[j] - 1.96 * se[j], -50, 50))), float(np.exp(np.clip(beta[j] + 1.96 * se[j], -50, 50)))],
                            'p': float(2 * norm.sf(abs(z))), 'p_boot': None if p_boot is None else float(p_boot[j]),
                            'separation': bool(abs(beta[j]) > 5)})     # > 5 log-odds per SD: (quasi-)separated
    if eidx:
        b, Vb = beta[eidx], V[np.ix_(eidx, eidx)]
        stat = float(b @ np.linalg.solve(Vb, b))
        out['wald_E'] = {'chi2': stat, 'df': len(eidx), 'p': float(chi2.sf(stat, len(eidx))),
                         'p_boot': None if pw_boot is None else float(pw_boot)}
    return out


# ---------------------------------------------------------------- report
def f3(v):
    return '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.3f}'


def fci(v, c, signed=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '–'
    fmt = '{:+.3f}' if signed else '{:.3f}'
    return f'{fmt.format(v)} [{fmt.format(c[0])}, {fmt.format(c[1])}]' if c and c[0] is not None else fmt.format(v)


def render(results, path):
    lines = ['# Escalation results', '',
             'The question: a depeg episode started L hours ago and has not reached -5 % (held for two hours) yet. '
             'Will it (severe), and will it last at least 24 h (long)? Episodes already severe at the decision hour '
             'are left out, and so are episodes still open at the end of the data whose answer is unknown.', '',
             '"time": fitted on episodes that started by 2025-09-30, scored on later ones. "LOAO": leave one asset '
             'out over all years (the asset is never seen in training, so exposure cannot act as an asset '
             'fingerprint). Brackets: 95 % intervals from an asset-cluster bootstrap. prior = random ranking.', '']
    order = {('severe', 1): 0, ('severe', 6): 1, ('long', 1): 2, ('long', 6): 3}
    for r in sorted(results, key=lambda r: (order.get((r['outcome'], r['landmark']), 9), r['landmark'])):
        t, lo = r['protocols'].get('time', {}), r['protocols'].get('loao', {})
        lines += [f"## {r['outcome'].capitalize()}, {r['landmark']} h after the start", '',
                  f"{r['n']} episodes from {r['assets']} assets, {r['positives']} positive ({100 * r['positives'] / r['n']:.1f}%); "
                  f"time-split test: {t.get('n', '–')} episodes, {t.get('positives', '–')} positive; "
                  f"left out as already decided: {r.get('decided', 0)}.", '',
                  '| model | time AP | time ROC-AUC | time Brier | LOAO AP | LOAO ROC-AUC | LOAO Brier |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for m in MODELS:
            a, b = t.get('models', {}).get(m), lo.get('models', {}).get(m)
            if not a and not b:
                continue
            a, b = a or {}, b or {}
            lines.append(f"| {m} | {fci(a.get('ap'), a.get('ap_ci'))} | {fci(a.get('roc_auc'), a.get('roc_auc_ci'))} | "
                         f"{f3(a.get('brier'))} | {fci(b.get('ap'), b.get('ap_ci'))} | "
                         f"{fci(b.get('roc_auc'), b.get('roc_auc_ci'))} | {f3(b.get('brier'))} |")
        lines += ['', '| difference | time ΔAP | time ΔROC-AUC | LOAO ΔAP | LOAO ΔROC-AUC |', '| --- | --- | --- | --- | --- |']
        for a_, b_ in PAIRS:
            key = f'{a_} - {b_}'
            x, z = t.get('pairs', {}).get(key), lo.get('pairs', {}).get(key)
            if not x and not z:
                continue
            x, z = x or {}, z or {}
            lines.append(f"| {key} | {fci(x.get('ap'), x.get('ap_ci'), True)} | {fci(x.get('roc_auc'), x.get('roc_auc_ci'), True)} | "
                         f"{fci(z.get('ap'), z.get('ap_ci'), True)} | {fci(z.get('roc_auc'), z.get('roc_auc_ci'), True)} |")
        lines.append('')
        titles = {'all': 'all episodes', 'exposed': 'episodes of assets with lending exposure',
                  'category': 'all episodes, with asset-category dummies'}
        for key, title in titles.items():
            inf = r.get('inference', {}).get(key)
            if not inf or 'coef' not in inf:
                continue
            w = inf.get('wald_E')
            shown = inf['coef'] if key == 'all' else [c for c in inf['coef'] if c['block'] == 'E']
            lines += [f"Logistic regression on {title} ({inf['n']} episodes, {inf['positives']} positive, "
                      f"{inf['clusters']} assets; standard errors clustered by asset"
                      f"{'' if inf['converged'] else '; did not converge'}"
                      f"{'' if key == 'all' else '; exposure rows only'}):", '',
                      '| feature | block | odds ratio per SD [95 % CI] | p | p (bootstrap-t) |', '| --- | --- | --- | --- | --- |']
            for c in shown:
                pb = '–' if c.get('p_boot') is None else f"{c['p_boot']:.3f}"
                lines.append(f"| {c['feature']}{' (separated)' if c.get('separation') else ''} | {c['block']} | "
                             f"{c['odds_ratio']:.2f} [{c['or_ci'][0]:.2f}, {c['or_ci'][1]:.2f}] | {c['p']:.3f} | {pb} |")
            if w:
                pb = '' if w.get('p_boot') is None else f"; cluster bootstrap p = {w['p_boot']:.3f} ({inf['bootstrap_draws']} draws)"
                lines.append(f"\nExposure block jointly: Wald chi2({w['df']}) = {w['chi2']:.2f}, p = {w['p']:.3f}{pb}.")
            if inf.get('dropped'):
                lines.append(f"Constant here, left out: {', '.join(inf['dropped'])}.")
            lines.append('')
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'model' / 'escalation.csv.gz'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'model'))
    ap.add_argument('--landmarks', nargs='+', type=int)
    ap.add_argument('--outcomes', nargs='+', default=list(OUTCOMES), choices=list(OUTCOMES))
    ap.add_argument('--protocols', nargs='+', default=list(PROTOCOLS), choices=list(PROTOCOLS))
    ap.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--inference-bootstrap', type=int, default=999, help='cluster bootstrap-t draws for the regressions (0 = off)')
    args = ap.parse_args()

    table = pd.read_csv(args.data)
    groups = json.loads((Path(args.data).parent / 'escalation_features.json').read_text())
    sets = {'hgb_P': groups['EP'] + groups['P'], 'hgb_PE': groups['EP'] + groups['P'] + groups['E']}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / 'escalation_results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    landmarks = args.landmarks or sorted(table['landmark'].unique())

    for L in landmarks:
        for outcome in args.outcomes:
            y = OUTCOMES[outcome]
            d = table[(table['landmark'] == L) & table[y].notna()].reset_index(drop=True)
            if d.empty or d[y].nunique() < 2:
                continue
            t0 = time.time()
            yy = d[y].to_numpy(dtype=float)
            entry = {'landmark': int(L), 'outcome': outcome, 'n': int(len(d)), 'positives': int(yy.sum()),
                     'assets': int(d['symbol'].nunique()),
                     'decided': int(table.loc[table['landmark'] == L, 'decided_severe'].sum()) if outcome == 'severe' else 0,
                     'protocols': {}, 'inference': {}}
            frames = []
            for protocol in args.protocols:
                mask, preds = run_protocol(d, y, outcome, args.models, sets, protocol)
                ys, gs = yy[mask], d.loc[mask, 'symbol'].to_numpy()
                ps = {m: p[mask] for m, p in preds.items()}
                boot = cluster_bootstrap(gs, ys, ps, args.bootstrap, args.seed)
                res = {'n': int(mask.sum()), 'positives': int(ys.sum()), 'assets': int(len(np.unique(gs))),
                       'models': {}, 'pairs': {}}
                for m, s in ps.items():
                    pm = point_metrics(ys, s, m in PROBABILISTIC)
                    pm.update({'ap_ci': ci(boot[m]['ap']), 'roc_auc_ci': ci(boot[m]['roc_auc'])})
                    res['models'][m] = pm
                for a_, b_ in PAIRS:
                    if a_ in ps and b_ in ps:
                        da, db = boot[a_], boot[b_]
                        res['pairs'][f'{a_} - {b_}'] = {
                            'ap': res['models'][a_]['ap'] - res['models'][b_]['ap'],
                            'ap_ci': ci(da['ap'] - db['ap']), 'ap_share_above_0': float(np.mean(da['ap'] - db['ap'] > 0)),
                            'roc_auc': res['models'][a_]['roc_auc'] - res['models'][b_]['roc_auc'],
                            'roc_auc_ci': ci(da['roc_auc'] - db['roc_auc']),
                            'roc_auc_share_above_0': float(np.mean(da['roc_auc'] - db['roc_auc'] > 0))}
                res['bootstrap_draws'] = int(len(next(iter(boot.values()))['ap']))
                entry['protocols'][protocol] = res
                for m, s in ps.items():
                    frames.append(pd.DataFrame({'episode_id': d.loc[mask, 'episode_id'].to_numpy(), 'symbol': gs,
                                                'protocol': protocol, 'model': m, 'y': ys.astype(int), 'score': s}))
            entry['inference'] = {spec: inference(d, y, outcome, spec, args.inference_bootstrap) for spec in ('all', 'exposed', 'category')}
            entry['seconds'] = round(time.time() - t0, 1)
            pd.concat(frames).to_csv(out / f'escalation_preds_{outcome}_{L}h.csv.gz', index=False, float_format='%.6g')
            results = [r for r in results if not (r['landmark'] == L and r['outcome'] == outcome)] + [entry]
            res_path.write_text(json.dumps(results, indent=1))
            msg = []
            for protocol, res in entry['protocols'].items():
                best = {m: res['models'][m]['ap'] for m in ('depth', 'logit_core', 'logit_coreE', 'hgb_P', 'hgb_PE')
                        if m in res['models']}
                msg.append(f"{protocol}: " + ', '.join(f'{m} {v:.3f}' for m, v in best.items()))
            print(f"{outcome:6s} L={L:2d}h  n={entry['n']} pos={entry['positives']}  AP  " + ' | '.join(msg) +
                  f"  ({entry['seconds']} s)")
    render(results, out / 'escalation_results.md')
    print(f'wrote {out / "escalation_results.md"}')


if __name__ == '__main__':
    main()
