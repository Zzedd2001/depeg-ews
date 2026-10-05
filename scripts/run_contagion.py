#!/usr/bin/env python3
"""Contagion ranking (RQ3): once asset s has depegged, rank the other assets by the chance that they
start an episode within the next 72 h / week. Reads data/model/contagion_<seeds>.csv.gz (make_contagion.py).

Rankers (fixed in advance):
  random       random order: the reference (precision@5 = share of followers among the candidates)
  hist         the candidate's episodes in the last 365 days ("usual suspects")
  stress       the candidate's deepest deviation of the last 24 h
  follow       how often the candidate followed other assets' episodes within a week in the past
  logit_own    logistic regression on the candidate's own features (OWN)
  logit_sim    + similarity to the seed without the graph (SIM)
  logit_fam    + same token family (FAM)
  logit_graph  + lending links (GRAPH: Morpho markets and vaults, Aave v3 / Spark pools)
  hgb_fam, hgb_graph   gradient-boosted trees on OWN + SIM + FAM, and on everything
Protocols:
  time   fit on seeds decided by 2025-09-30, rank the candidates of later seeds (as deployed)
  loco   leave one seed cluster out (seeds < 72 h apart share a cluster), over all seeds
Metrics: per seed with at least one follower, AP of the ranking, precision@5, hit@5 (a follower in the
top 5) and NDCG@10, averaged over seeds; AP and ROC-AUC over all pairs pooled. 95 % intervals by
resampling seed clusters, with paired differences on the same resamples.
Inference: logistic regression of following on all features (standardised), standard errors clustered
by seed cluster, p-values also from the pairs cluster bootstrap-t (run_escalation.py); joint tests of
GRAPH and of its Morpho and Aave / Spark parts.
Case studies: followers and the top 5 under logit_fam and logit_graph for every collapse in the test period.

Usage:
  python scripts/run_contagion.py                                 # severe seeds, 72 h and 168 h
  python scripts/run_contagion.py --seeds all --windows 168
Writes data/model/contagion_<seeds>_results.json / .md and contagion_<seeds>_preds.csv.gz.
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_escalation as rx  # noqa: E402
from common import ROOT, winsor_bounds  # noqa: E402
from compare_preds import SortedAP  # noqa: E402

MODELS = ['random', 'hist', 'stress', 'follow', 'logit_own', 'logit_sim', 'logit_fam', 'logit_graph', 'hgb_fam', 'hgb_graph']
LEARNED = {'logit_own': ['OWN'], 'logit_sim': ['OWN', 'SIM'], 'logit_fam': ['OWN', 'SIM', 'FAM'],
           'logit_graph': ['OWN', 'SIM', 'FAM', 'GRAPH'], 'hgb_fam': ['OWN', 'SIM', 'FAM'],
           'hgb_graph': ['OWN', 'SIM', 'FAM', 'GRAPH']}
PAIRS = [('logit_sim', 'logit_own'), ('logit_fam', 'logit_sim'), ('logit_graph', 'logit_fam'), ('hgb_graph', 'hgb_fam'),
         ('logit_own', 'hist')]
METRICS = ['ap', 'p5', 'hit5', 'ndcg10']
RANDOM_DRAWS = 200
SOURCES = {'morpho': 'GRAPH_MORPHO', 'pool': 'GRAPH_POOL'}      # parts of GRAPH (older feature files have GRAPH only)


def source_cols(groups, src):
    if src == 'morpho':
        return groups.get('GRAPH_MORPHO', groups['GRAPH'])
    return groups.get(SOURCES[src], [])


# ---------------------------------------------------------------- rankers
def fit_logit(tr, te, y, cols, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    a, b = tr[cols].astype(float), te[cols].astype(float)
    (lo, hi), med = winsor_bounds(a, 0.01, 0.99), a.median()
    a = a.clip(lo, hi, axis=1).fillna(med).fillna(0.0)
    b = b.clip(lo, hi, axis=1).fillna(med).fillna(0.0)
    sc = StandardScaler().fit(a)
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(a), tr[y].to_numpy())
    return m.predict_proba(sc.transform(b))[:, 1]


def score(name, tr, te, y, groups, seed=0):
    if name == 'random':
        return np.random.default_rng(seed).uniform(size=len(te))       # metrics for 'random' are averaged over draws
    if name == 'hist':
        return te['episodes_365d'].fillna(0).to_numpy(dtype=float)
    if name == 'stress':
        return -te['z_min_24h'].fillna(0).to_numpy(dtype=float)
    if name == 'follow':
        return te['follow_rate'].to_numpy(dtype=float)
    cols = sum((groups[g] for g in LEARNED[name]), [])
    if tr[y].nunique() < 2:
        return np.full(len(te), tr[y].mean())
    if name.startswith('logit'):
        return fit_logit(tr, te, y, cols)
    return rx.fit_hgb(tr, te, y, cols)


def predict(d, y, models, groups, protocol):
    """Out-of-sample scores; returns (mask of scored rows, {model: scores})."""
    preds = {m: np.full(len(d), np.nan) for m in models}
    if protocol == 'time':
        fit = (d['split'] == 'train').to_numpy()
        for m in models:
            preds[m][~fit] = score(m, d[fit], d[~fit], y, groups)
        return ~fit, preds
    for c in sorted(d['cluster'].unique()):
        test = (d['cluster'] == c).to_numpy()
        for m in models:
            preds[m][test] = score(m, d[~test], d[test], y, groups, seed=int(c))
    return np.ones(len(d), bool), preds


# ---------------------------------------------------------------- metrics
def jitter(seed_ids, cands):
    """Deterministic tie-breaker far below any score difference."""
    return np.array([zlib.crc32(f'{s}|{c}'.encode()) for s, c in zip(seed_ids, cands)], dtype=float) / 2 ** 32 * 1e-9


def seed_metrics(y, s, k=5, n=10):
    order = np.argsort(-s, kind='mergesort')
    yy = y[order]
    disc = 1 / np.log2(np.arange(2, n + 2))
    top = yy[:k]
    idcg = disc[:int(min(yy.sum(), n))].sum()
    return {'ap': rx.ap_score(y, s), 'p5': float(top.sum() / min(k, len(yy))), 'hit5': float(top.sum() > 0),
            'ndcg10': float((yy[:n] * disc[:len(yy[:n])]).sum() / idcg)}


def per_seed(frame, y, preds, models, draws=RANDOM_DRAWS, seed=0):
    """frame: the scored rows (seed_id, cand, cluster, y); returns one row per seed with followers."""
    rng = np.random.default_rng(seed)
    rows = []
    jit = jitter(frame['seed_id'], frame['cand'])
    for sid, idx in frame.groupby('seed_id', sort=False).indices.items():
        yy = frame[y].to_numpy()[idx]
        if yy.sum() == 0:
            continue
        r = {'seed_id': sid, 'cluster': int(frame['cluster'].to_numpy()[idx[0]]), 'candidates': len(idx), 'followers': int(yy.sum())}
        for m in models:
            if m == 'random':
                ms = [seed_metrics(yy, rng.uniform(size=len(idx))) for _ in range(draws)]
                vals = {k: float(np.mean([x[k] for x in ms])) for k in METRICS}
            else:
                vals = seed_metrics(yy, preds[m][idx] + jit[idx])
            for k in METRICS:
                r[f'{m}:{k}'] = vals[k]
        rows.append(r)
    return pd.DataFrame(rows)


def bootstrap(ps, frame, y, preds, models, B, seed=0):
    """Resample seed clusters; returns means, intervals and paired differences of the per-seed metrics,
    and pooled AP with its interval."""
    rng = np.random.default_rng(seed)
    clusters = np.unique(frame['cluster'])
    pos = {c: i for i, c in enumerate(clusters)}
    counts = np.stack([np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)) for _ in range(B)])
    w_seed = counts[:, ps['cluster'].map(pos).to_numpy()].astype(float)            # B x seeds
    keep = w_seed.sum(axis=1) > 0
    w_seed = w_seed[keep]
    out = {'models': {}, 'pairs': {}, 'draws': int(keep.sum())}
    yy = frame[y].to_numpy(dtype=float)
    row_pos = frame['cluster'].map(pos).to_numpy()
    pooled_counts = counts[keep][:min(int(keep.sum()), 500)]                 # 500 draws are plenty for the pooled AP
    for m in models:
        res = {}
        for k in METRICS:
            v = ps[f'{m}:{k}'].to_numpy()
            draws = (w_seed @ v) / w_seed.sum(axis=1)
            res[k] = float(v.mean())
            res[f'{k}_ci'] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
        if m != 'random':
            sap = SortedAP(yy, preds[m])
            res['pooled_ap'] = sap(np.ones(len(yy)))
            pd_draws = np.array([sap(c[row_pos].astype(float)) for c in pooled_counts])
            res['pooled_ap_ci'] = [float(np.nanquantile(pd_draws, 0.025)), float(np.nanquantile(pd_draws, 0.975))]
            res['pooled_auc'] = rx.auc_score(yy, preds[m])
        out['models'][m] = res
    for a, b in PAIRS:
        if a in models and b in models:
            res = {}
            for k in ('ap', 'p5'):
                v = ps[f'{a}:{k}'].to_numpy() - ps[f'{b}:{k}'].to_numpy()
                draws = (w_seed @ v) / w_seed.sum(axis=1)
                res[k] = float(v.mean())
                res[f'{k}_ci'] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
                res[f'{k}_share_above_0'] = float(np.mean(draws > 0))
            out['pairs'][f'{a} - {b}'] = res
    return out


# ---------------------------------------------------------------- inference
def inference(d, y, groups, boot):
    from scipy.stats import chi2, norm
    cols = groups['OWN'] + groups['SIM'] + groups['FAM'] + groups['GRAPH']
    X = d[cols].astype(float)
    X = X.clip(*winsor_bounds(X, 0.01, 0.99), axis=1).fillna(X.median()).fillna(0.0)
    X = X.loc[:, X.std() > 1e-9]
    corr = X.corr().abs().to_numpy()
    keep = []                                            # drop columns (almost) collinear with an earlier one
    for i in range(len(X.columns)):
        if all(corr[i, k] < 0.995 for k in keep):
            keep.append(i)
    X = X.iloc[:, keep]
    names = list(X.columns)
    yy = d[y].to_numpy(dtype=float)
    out = {'n': int(len(d)), 'positives': int(yy.sum()), 'clusters': int(d['cluster'].nunique()),
           'dropped': [c for c in cols if c not in names], 'coef': []}
    Xs = ((X - X.mean()) / X.std()).to_numpy()
    g = d['cluster'].to_numpy()
    beta, V, conv, G = rx.logit_clustered(Xs, yy, g)
    se = np.sqrt(np.diag(V))
    sets = {'graph': [j for j, c in enumerate(names, start=1) if c in groups['GRAPH']]}
    if 'GRAPH_POOL' in groups:                     # the Morpho and the Aave / Spark parts on their own
        for src in SOURCES:
            sets[src] = [j for j, c in enumerate(names, start=1) if c in source_cols(groups, src)]
    sets = {k: v for k, v in sets.items() if v}
    p_boot, pw_boot, done = rx.bootstrap_t_sets(Xs, yy, g, beta, V, sets, boot) if boot else (None, {}, 0)
    out.update({'converged': conv, 'bootstrap_draws': done})
    for j, c in enumerate(names, start=1):
        block = next(b for b in ('OWN', 'SIM', 'FAM', 'GRAPH') if c in groups[b])
        src = next((s for s in SOURCES if c in source_cols(groups, s)), None) if block == 'GRAPH' else None
        out['coef'].append({'feature': c, 'block': block, 'source': src, 'odds_ratio': float(np.exp(np.clip(beta[j], -50, 50))),
                            'or_ci': [float(np.exp(np.clip(beta[j] - 1.96 * se[j], -50, 50))), float(np.exp(np.clip(beta[j] + 1.96 * se[j], -50, 50)))],
                            'p': float(2 * norm.sf(abs(beta[j] / se[j]))), 'p_boot': None if p_boot is None else float(p_boot[j]),
                            'separation': bool(abs(beta[j]) > 5)})
    for name, idx in sets.items():
        b, Vb = beta[idx], V[np.ix_(idx, idx)]
        stat = float(b @ np.linalg.solve(Vb, b))
        out[f'wald_{name}'] = {'chi2': stat, 'df': len(idx), 'p': float(chi2.sf(stat, len(idx))),
                               'p_boot': None if name not in pw_boot else float(pw_boot[name])}
    return out


def link_rates(d, y, linked, B=1000, seed=0):
    """Follow rate of pairs with and without a link, and the difference with a cluster bootstrap."""
    yy = d[y].to_numpy(dtype=float)
    out = {'linked_pairs': int(linked.sum()), 'linked_followers': int(yy[linked].sum()),
           'rate_linked': float(yy[linked].mean()) if linked.any() else None, 'rate_unlinked': float(yy[~linked].mean())}
    if linked.any() and (~linked).any():
        rng = np.random.default_rng(seed)
        clusters = np.unique(d['cluster'])
        pos = d['cluster'].map({c: i for i, c in enumerate(clusters)}).to_numpy()
        diffs = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))[pos].astype(float)
            a, b = w[linked], w[~linked]
            if a.sum() > 0 and b.sum() > 0:
                diffs.append((a @ yy[linked]) / a.sum() - (b @ yy[~linked]) / b.sum())
        out['difference'] = out['rate_linked'] - out['rate_unlinked']
        out['difference_ci'] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    return out


def links(d, y, groups, B=1000, seed=0):
    """link_rates for any GRAPH link (top level) and for Morpho and Aave / Spark links on their own ('by_source');
    unlinked pairs include same-family pairs (their GRAPH features are zero)."""
    out = link_rates(d, y, d[groups['GRAPH']].gt(0).any(axis=1).to_numpy(), B, seed)
    yy = d[y].to_numpy(dtype=float)
    fam = d['same_family'].to_numpy() > 0
    out.update({'same_family_pairs': int(fam.sum()), 'rate_same_family': float(yy[fam].mean()) if fam.any() else None,
                'by_source': {}})
    for src in SOURCES:
        cols = source_cols(groups, src)
        if cols and 'GRAPH_POOL' in groups:
            out['by_source'][src] = link_rates(d, y, d[cols].gt(0).any(axis=1).to_numpy(), B, seed)
    return out


def case_studies(frame, y, preds, groups, k=5):
    rows = []
    test = frame.assign(_fam=preds['logit_fam'], _graph=preds['logit_graph'])
    linked = lambda g, cols: ', '.join(sorted(g.loc[g[cols].gt(0).any(axis=1), 'cand'])) or '–' if cols else '–'
    for sid, g in test[test['seed_severity'] == 'collapse'].groupby('seed_id', sort=False):
        top = lambda col: ', '.join(f"{c}{'*' if v == 1 else ''}" for c, v in g.sort_values(col, ascending=False)[['cand', y]].head(k).itertuples(index=False))
        rows.append({'seed': g['seed'].iat[0], 'time': g['seed_time'].iat[0], 'followers': ', '.join(sorted(g.loc[g[y] == 1, 'cand'])) or '–',
                     'top_fam': top('_fam'), 'top_graph': top('_graph'),
                     'linked': linked(g, source_cols(groups, 'morpho')), 'pool_linked': linked(g, source_cols(groups, 'pool'))})
    return rows


# ---------------------------------------------------------------- report
def f3(v):
    return '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.3f}'


def fci(v, c, signed=False):
    if v is None:
        return '–'
    fmt = '{:+.3f}' if signed else '{:.3f}'
    return f'{fmt.format(v)} [{fmt.format(c[0])}, {fmt.format(c[1])}]'


def render(results, path, kind):
    lines = [f'# Contagion results ({kind} seeds)', '',
             'Once an asset has depegged (severe seeds: when its held depth first reached -5 %), rank every other asset '
             'that is not already in an episode by the chance that it starts one within the window. Metrics per seed with '
             'at least one follower, averaged: AP of the ranking, precision@5, hit@5 (a follower in the top 5), NDCG@10; '
             'pooled AP over all pairs. "time": fitted on seeds decided by 2025-09-30, scored on later ones; "LOCO": leave '
             'one seed cluster out (seeds < 72 h apart). Brackets: 95 % intervals from resampling seed clusters.', '']
    for r in sorted(results, key=lambda r: r['window']):
        t, lo, lk = r['protocols'].get('time', {}), r['protocols'].get('loco', {}), r.get('links', {})
        lines += [f"## Followers within {r['window']} h", '',
                  f"{r['seeds']} seeds in {r['clusters']} clusters, {r['pairs']:,} seed-candidate pairs, {r['positives']} followed "
                  f"({100 * r['positives'] / r['pairs']:.1f}%); seeds with at least one follower: time-split test {t.get('seeds_scored', '–')}, "
                  f"LOCO {lo.get('seeds_scored', '–')}.", '',
                  f"Lending links: {lk.get('linked_pairs', 0)} pairs have any link ({lk.get('linked_followers', 0)} followed; "
                  f"rate {f3(lk.get('rate_linked'))} vs {f3(lk.get('rate_unlinked'))} without a link"
                  + (f", difference {fci(lk.get('difference'), lk.get('difference_ci'), True)}" if lk.get('difference_ci') else '')
                  + ')' + ''.join(
                      f"; {label} {s.get('linked_pairs', 0)} ({s.get('linked_followers', 0)} followed, rate {f3(s.get('rate_linked'))} vs "
                      f"{f3(s.get('rate_unlinked'))}" + (f", difference {fci(s.get('difference'), s.get('difference_ci'), True)}"
                                                         if s.get('difference_ci') else '') + ')'
                      for src, label in (('morpho', 'Morpho market / vault links'), ('pool', 'Aave / Spark pool links'))
                      for s in [lk.get('by_source', {}).get(src)] if s)
                  + f". Same-family pairs: {lk.get('same_family_pairs', 0)} (rate {f3(lk.get('rate_same_family'))}).", '',
                  '| ranker | time AP | time P@5 | time hit@5 | time NDCG@10 | time pooled AP | LOCO AP | LOCO P@5 | LOCO pooled AP |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for m in MODELS:
            a, b = t.get('models', {}).get(m), lo.get('models', {}).get(m)
            if not a and not b:
                continue
            a, b = a or {}, b or {}
            lines.append(f"| {m} | {fci(a.get('ap'), a.get('ap_ci'))} | {fci(a.get('p5'), a.get('p5_ci'))} | {f3(a.get('hit5'))} | "
                         f"{f3(a.get('ndcg10'))} | {f3(a.get('pooled_ap'))} | {fci(b.get('ap'), b.get('ap_ci'))} | "
                         f"{fci(b.get('p5'), b.get('p5_ci'))} | {f3(b.get('pooled_ap'))} |")
        lines += ['', '| difference | time ΔAP | time ΔP@5 | LOCO ΔAP | LOCO ΔP@5 |', '| --- | --- | --- | --- | --- |']
        for a_, b_ in PAIRS:
            key = f'{a_} - {b_}'
            x, z = t.get('pairs', {}).get(key), lo.get('pairs', {}).get(key)
            if not x and not z:
                continue
            x, z = x or {}, z or {}
            lines.append(f"| {key} | {fci(x.get('ap'), x.get('ap_ci'), True)} | {fci(x.get('p5'), x.get('p5_ci'), True)} | "
                         f"{fci(z.get('ap'), z.get('ap_ci'), True)} | {fci(z.get('p5'), z.get('p5_ci'), True)} |")
        inf = r.get('inference')
        if inf and inf.get('coef'):
            lines += ['', f"Logistic regression on all pairs ({inf['n']:,} pairs, {inf['positives']} followed, {inf['clusters']} seed clusters; "
                      'SE clustered by seed cluster):', '', '| feature | block | odds ratio per SD [95 % CI] | p | p (bootstrap-t) |',
                      '| --- | --- | --- | --- | --- |']
            for c in inf['coef']:
                pb = '–' if c.get('p_boot') is None else f"{c['p_boot']:.3f}"
                blk = c['block'] + (f" ({'Aave / Spark' if c['source'] == 'pool' else 'Morpho'})" if c.get('source') else '')
                lines.append(f"| {c['feature']}{' (separated)' if c['separation'] else ''} | {blk} | {c['odds_ratio']:.2f} "
                             f"[{c['or_ci'][0]:.2f}, {c['or_ci'][1]:.2f}] | {c['p']:.3f} | {pb} |")
            first = True
            for key, label in (('wald_graph', 'GRAPH block jointly'), ('wald_morpho', 'Morpho part'), ('wald_pool', 'Aave / Spark part')):
                w = inf.get(key)
                if w:
                    pb = '' if w.get('p_boot') is None else f"; cluster bootstrap p = {w['p_boot']:.3f}"
                    lines.append(f"{chr(10) if first else ''}{label}: Wald chi2({w['df']}) = {w['chi2']:.2f}, p = {w['p']:.3f}{pb}.")
                    first = False
            if inf.get('dropped'):
                lines.append(f"Constant or collinear with another feature here, left out: {', '.join(inf['dropped'])}.")
        cs = r.get('cases')
        if cs:
            lines += ['', 'Collapses in the test period (time split; * = followed within the window):', '',
                      '| seed | decided | followers | top 5, logit_fam | top 5, logit_graph | candidates with a Morpho link | with an Aave / Spark pool link |',
                      '| --- | --- | --- | --- | --- | --- | --- |']
            for c in cs:
                lines.append(f"| {c['seed']} | {c['time'][:13]} | {c['followers']} | {c['top_fam']} | {c['top_graph']} | {c['linked']} | "
                             f"{c.get('pool_linked', '–')} |")
        lines.append('')
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', choices=['severe', 'all'], default='severe')
    ap.add_argument('--data', help='default: data/model/contagion_<seeds>.csv.gz')
    ap.add_argument('--out', default=str(ROOT / 'data' / 'model'))
    ap.add_argument('--windows', nargs='+', type=int, default=[72, 168])
    ap.add_argument('--protocols', nargs='+', default=['time', 'loco'], choices=['time', 'loco'])
    ap.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--inference-bootstrap', type=int, default=499)
    args = ap.parse_args()

    data = Path(args.data or ROOT / 'data' / 'model' / f'contagion_{args.seeds}.csv.gz')
    table = pd.read_csv(data)
    groups = json.loads((data.parent / 'contagion_features.json').read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / f'contagion_{args.seeds}_results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    frames = []
    for w in args.windows:
        y = f'y{w}'
        d = table[table[y].notna()].reset_index(drop=True)
        t0 = time.time()
        entry = {'window': w, 'seeds': int(d['seed_id'].nunique()), 'clusters': int(d['cluster'].nunique()), 'pairs': int(len(d)),
                 'positives': int(d[y].sum()), 'protocols': {}, 'links': links(d, y, groups),
                 'inference': inference(d, y, groups, args.inference_bootstrap)}
        for protocol in args.protocols:
            mask, preds = predict(d, y, args.models, groups, protocol)
            frame = d[mask].reset_index(drop=True)
            p = {m: v[mask] for m, v in preds.items()}
            ps = per_seed(frame, y, p, args.models)
            if ps.empty:
                print(f'{protocol}: no scored seed has a follower within {w} h; skipped')
                continue
            res = bootstrap(ps, frame, y, p, args.models, args.bootstrap)
            res.update({'seeds_scored': int(len(ps)), 'pairs_scored': int(len(frame))})
            entry['protocols'][protocol] = res
            if protocol == 'time' and {'logit_fam', 'logit_graph'} <= set(args.models):
                entry['cases'] = case_studies(frame, y, p, groups)
            for m in args.models:
                frames.append(pd.DataFrame({'window': w, 'protocol': protocol, 'model': m, 'seed_id': frame['seed_id'],
                                            'cand': frame['cand'], 'y': frame[y].astype(int), 'score': p[m]}))
        entry['seconds'] = round(time.time() - t0, 1)
        old = next((r for r in results if r['window'] == w), None)
        if old is not None:                               # a run with other protocols / models keeps what it did not redo
            for protocol, res in entry['protocols'].items():
                prev = old.get('protocols', {}).get(protocol)
                if prev and set(prev.get('models', {})) - set(res['models']):
                    res['models'] = {**{m: v for m, v in prev['models'].items() if m not in res['models']}, **res['models']}
                    res['pairs'] = {**prev.get('pairs', {}), **res['pairs']}
            entry['protocols'] = {**old.get('protocols', {}), **entry['protocols']}
            entry.setdefault('cases', old.get('cases'))
            if not args.inference_bootstrap and old.get('inference'):
                entry['inference'] = old['inference']
        results = [r for r in results if r['window'] != w] + [entry]
        res_path.write_text(json.dumps(results, indent=1))
        tt = entry['protocols'].get('time', {}).get('models', {})
        print(f"{args.seeds} seeds, {w:3d} h: {entry['seeds']} seeds, {entry['pairs']} pairs, {entry['positives']} followed; time AP "
              + ', '.join(f"{m} {tt[m]['ap']:.3f}" for m in ('random', 'hist', 'logit_own', 'logit_fam', 'logit_graph') if m in tt)
              + f"  ({entry['seconds']} s)")
    if frames:
        new = pd.concat(frames)
        path = out / f'contagion_{args.seeds}_preds.csv.gz'
        if path.exists():                                 # keep earlier runs' other windows / protocols / models
            prev = pd.read_csv(path)
            key = ['window', 'protocol', 'model']
            done = new[key].drop_duplicates()
            prev = prev.merge(done, on=key, how='left', indicator=True)
            new = pd.concat([prev[prev['_merge'] == 'left_only'].drop(columns='_merge'), new], ignore_index=True)
        new.to_csv(path, index=False, float_format='%.6g')
    render(results, out / f'contagion_{args.seeds}_results.md', args.seeds)
    print(f"wrote {out / f'contagion_{args.seeds}_results.md'}")


if __name__ == '__main__':
    main()
