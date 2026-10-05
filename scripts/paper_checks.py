#!/usr/bin/env python3
"""Numbers the paper quotes that the main scripts do not print. Run after the pipeline (README).

  python scripts/paper_checks.py repeats      # 6.1: 24-h recall on repeat episodes and on the others
  python scripts/paper_checks.py horizon      # 6.1, Fig. 4(a): episodes flagged a day / two days ahead; hours at
                                              #      which the 72-h models overtake the deviation rule
  python scripts/paper_checks.py waits        # 6.2, 7: hours until -5% holds; deepest fifth caught and missed;
                                              #      a depth cut set on earlier episodes
  python scripts/paper_checks.py monitor      # 7: validation-threshold alerts per day
  python scripts/paper_checks.py chance       # 6.1, 7: what random alerts at the same rate would catch (expected values)
  python scripts/paper_checks.py drift        # 6.4: how far exposure levels move between training and test
  python scripts/paper_checks.py treeseeds    # 5, 8: single-seed onset trees (the published ones average five), ~10 min
  python scripts/paper_checks.py seedset      # 8: the five-seed onset trees refitted on seeds 5-9, ~20 min
  python scripts/paper_checks.py seeds        # 6.3: per-seed AP changes from the family indicator and the links
  python scripts/paper_checks.py separation   # 6.5: links nonzero only for pairs that never follow; the Wald
                                              #      statistic along the direction in which their estimates diverge
  python scripts/paper_checks.py shock        # 6.4: semi-synthetic replicates as low as the observed follow-rate
                                              #      gap when each seed's candidates share a common shock
  python scripts/paper_checks.py drop --data data/model/escalation.csv.gz --outcome long --landmark 6 \\
      --spec exposed --feature dlog_family_borrow_against_usd_24h          # 6.4: drop one asset at a time
  python scripts/paper_checks.py drop --data data/model_robust/escalation.csv.gz --landmark 1 \\
      --feature mm_blind_oracle_share                                      # 6.5 (and --landmark 6); also --spec all
                                                                           #      --landmark 6 with --feature
                                                                           #      family_borrow_to_collateral or
                                                                           #      dlog_family_borrow_against_usd_24h
  python scripts/paper_checks.py synthetic --p-link 0.7 --p-other 0.04 --seed 0   # 6.4: planted vault channel
  python scripts/paper_checks.py synthetic --p-link 0.06 --p-other 0.06 --seed 1  #      replicate without it

Every subcommand reads data/ under --root (default: the repository root). Prediction files are read with
float_precision='round_trip', which returns exactly the scores the models wrote.
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, epoch_seconds, not_suspect, winsor_bounds  # noqa: E402

H = 3600


def read_preds(path):
    return pd.read_csv(path, float_precision='round_trip')


def episodes(root, labels='labels'):
    ep = not_suspect(pd.read_csv(root / 'data' / labels / 'episodes.csv'))
    ep['ts'], ep['te'] = epoch_seconds(ep['start']), epoch_seconds(ep['end'])
    return ep


def observed_dev(root):
    lab = pd.read_csv(root / 'data' / 'labels' / 'labels_hourly.csv.gz', usecols=['hour', 'symbol', 'dev', 'obs'])
    lab['ts'] = epoch_seconds(lab['hour'])
    return {s: g.set_index('ts')['dev'].where(g.set_index('ts')['obs'] == 1).sort_index() for s, g in lab.groupby('symbol')}


def top_alerts(p, budget=0.05):
    """Exactly the top round(budget * N) scores, ties broken by time order (as run_baselines does)."""
    top = np.argsort(-p['score'].to_numpy(), kind='stable')[:int(round(budget * len(p)))]
    return p.iloc[np.sort(top)]


# ---------------------------------------------------------------- onset
def repeats(args):
    """Episodes that start within a week (168 h) of the end of the same asset's previous episode, and the rest."""
    ep = episodes(args.root)
    for m in ('hgb_P', 'logit_P', 'rule_dev'):
        p = read_preds(args.root / 'data' / 'model' / f'preds_{m}_24h.csv.gz')
        al = top_alerts(p)
        lo, hi = p['ts'].min(), p['ts'].max()
        e = ep[(ep['ts'] > lo) & (ep['ts'] <= hi + 24 * H)].copy()
        by = {s: g['ts'].to_numpy() for s, g in al.groupby('symbol')}
        e['caught'] = [bool(((by.get(s, np.array([])) >= t - 24 * H) & (by.get(s, np.array([])) <= t - H)).any())
                       for s, t in zip(e['symbol'], e['ts'])]
        prev = [ep.loc[(ep['symbol'] == s) & (ep['ts'] < t), 'te'].max() for s, t in zip(e['symbol'], e['ts'])]
        e['repeat'] = (e['ts'] - np.array(prev, dtype=float)) <= 7 * 24 * H
        r, o = e[e['repeat']], e[~e['repeat']]
        print(f"{m}: {len(e)} test episodes; repeats {len(r)}, caught {int(r['caught'].sum())}; "
              f"others {len(o)}, caught {int(o['caught'].sum())}")


def horizon(args):
    import make_figures as mf
    ep = episodes(args.root)
    share = {}
    for m in ('hgb_P', 'logit_P', 'rule_dev', 'base_rate'):
        k, s, n = mf.onset_horizon(args.root / 'data' / 'model', ep, m)
        share[m] = s
        print(f'{m}: {n} episodes; flagged at least 24 h ahead {100 * s[23]:.1f}%, 48 h ahead {100 * s[47]:.1f}%')
    for m in ('logit_P', 'hgb_P'):
        d = share[m] - share['rule_dev']
        stay = next((k for k in range(1, 73) if (d[k - 1:] > 0).all()), None)
        print(f'{m} above the deviation rule from {stay} h on (ties at {[k for k in range(1, 73) if d[k - 1] == 0]})')


def chance(args):
    """What alerts on K test asset-hours drawn at random would give, K being the alert count of the top-5% rule or
    of the validation threshold, in expectation and without simulation: an episode whose warning window holds w
    test rows is flagged with probability 1 - C(N - w, K) / C(N, K). Random alerts have the share of positive
    hours as precision and are almost all isolated hours. Compared with the price trees (Sections 6.1 and 7)."""
    from scipy.special import gammaln
    import make_figures as mf
    import run_baselines as rb
    ep = episodes(args.root)

    def p_hit(n, k, w):
        w = np.asarray(w, dtype=float)
        miss = np.exp(gammaln(n - w + 1) - gammaln(n - w - k + 1) - gammaln(n + 1) + gammaln(n - k + 1))
        return np.where(w > 0, 1 - miss, 0.0)

    def runs(al):
        al = al.sort_values(['symbol', 'ts'])
        return int((al.groupby('symbol')['ts'].diff().fillna(np.inf) > H).sum())

    res = json.loads((args.root / 'data' / 'model' / 'results.json').read_text())
    for h in (24, 72):
        p = read_preds(args.root / 'data' / 'model' / f'preds_hgb_P_{h}h.csv.gz')
        n, ts, by = len(p), p['ts'].to_numpy(), p.groupby('symbol').indices
        days = pd.to_datetime(p['ts'], unit='s', utc=True).dt.floor('D').nunique()
        adjacent = int((p.sort_values(['symbol', 'ts']).groupby('symbol')['ts'].diff() == H).sum())
        e = rb.split_episodes(ep, p)
        w, w_lead, severe = [], {k: [] for k in (24, 48)}, []
        for s, t0, sev in zip(e['symbol'], e['ts'], e['severity']):
            t = ts[by[s]] if s in by else np.array([])
            win = (t >= t0 - h * H) & (t <= t0 - H)
            if win.any():                                       # the episodes run_baselines.evaluate counts
                w.append(int(win.sum()))
                severe.append(sev in ('major', 'collapse'))
            for k in w_lead:
                w_lead[k].append(int(((t >= t0 - h * H) & (t <= t0 - k * H)).sum()))
        w, severe = np.array(w), np.array(severe)
        tree = next(r for r in res if r['model'] == 'hgb_P' and r['horizon'] == h)['test']
        rules = [('top 5%', int(round(0.05 * n)), tree['rate_0.05'], top_alerts(p))]
        if h == 24:
            thr = tree['budget_0.05']['threshold']
            rules.append(('validation threshold', int((p['score'] >= thr).sum()), tree['budget_0.05'], p[p['score'] >= thr]))
        for name, k, t_res, al in rules:
            hit = p_hit(n, k, w)
            r_runs = k - adjacent * k * (k - 1) / (n * (n - 1))
            print(f"{h} h, {name}: {k:,} of {n:,} asset-hours ({k / n:.2%}). Random alerts would precede "
                  f"{hit.mean():.2%} of the {len(w)} test episodes ({hit[severe].mean():.2%} of {int(severe.sum())} severe) "
                  f"at a precision of {tree['prevalence']:.2%}, in {r_runs / days:.1f} runs a day; the price trees precede "
                  f"{t_res['event_recall']:.1%} ({t_res['severe_recall']:.1%} of severe) at a precision of "
                  f"{t_res['alert_precision']:.2%}, in {runs(al) / days:.1f} runs a day")
        if h == 72:
            shares = {k: p_hit(n, int(round(0.05 * n)), w_lead[k]).mean() for k in w_lead}
            _, s_tree, n_ep = mf.onset_horizon(args.root / 'data' / 'model', ep, 'hgb_P')
            print(f"72 h, top 5%, flagged at least 24 h / 48 h ahead (Fig. 4(a), {n_ep} episodes): random "
                  f"{shares[24]:.2%} / {shares[48]:.2%}; price trees {s_tree[23]:.1%} / {s_tree[47]:.1%}")


def monitor(args):
    res = json.loads((args.root / 'data' / 'model' / 'results.json').read_text())
    thr = next(r for r in res if r['model'] == 'hgb_P' and r['horizon'] == 24)['test']['budget_0.05']['threshold']
    p = read_preds(args.root / 'data' / 'model' / 'preds_hgb_P_24h.csv.gz')
    al = p[p['score'] >= thr].sort_values(['symbol', 'ts'])
    days = pd.to_datetime(p['ts'], unit='s', utc=True).dt.floor('D').nunique()
    runs = int((al.groupby('symbol')['ts'].diff().fillna(np.inf) > H).sum())          # a new run of alerted hours
    day = pd.to_datetime(al['ts'], unit='s', utc=True).dt.floor('D')
    per_day = al.assign(d=day).groupby('d')['symbol'].nunique().sum() / days
    print(f'{len(al)} alerted asset-hours ({len(al) / len(p):.2%} of test hours) over {days} days = '
          f'{len(al) / days:.1f} a day; {runs} new runs of alerts = {runs / days:.1f} a day; {per_day:.1f} assets a day')


# ---------------------------------------------------------------- escalation
def waits(args):
    et = pd.read_csv(args.root / 'data' / 'model' / 'escalation.csv.gz')
    l1 = et[(et['landmark'] == 1) & et['y_severe'].notna()]
    te, pre = l1[l1['split'] == 'test'], l1[l1['split'] != 'test']
    k = int(round(0.2 * len(te)))
    top = te.sort_values('depth_frac', ascending=False, kind='stable').head(k)
    cut = pre['depth_frac'].quantile(0.8)
    flag = te['depth_frac'] >= cut
    print(f"test: {len(te)} episodes, {int(te['y_severe'].sum())} escalate; deepest fifth ({k}) catches "
          f"{int(top['y_severe'].sum())} (precision {top['y_severe'].mean():.3f}); cut at the 80th percentile of earlier "
          f"episodes ({cut:.3f} of 5%): flags {int(flag.sum())}, catches {int(te.loc[flag, 'y_severe'].sum())}")
    obs = observed_dev(args.root)
    pos = l1[l1['y_severe'] == 1]
    w = []
    for s, t0, tau, sp in zip(pos['symbol'], pos['t0'], pos['ts'], pos['split']):
        x = obs[s].loc[t0:].dropna()
        v, ix = x.to_numpy(), x.index.to_numpy()
        hit = next((ix[i] for i in range(1, len(v)) if max(v[i - 1], v[i]) <= -0.05), None)
        if hit is not None:
            w.append((s, t0, sp, (hit - tau) / H))
    w = pd.DataFrame(w, columns=['symbol', 't0', 'split', 'wait'])
    ids = set(zip(top['symbol'], top['t0']))
    w['caught'] = [(s, t) in ids for s, t in zip(w['symbol'], w['t0'])]
    tw = w[w['split'] == 'test']
    c, m = tw[tw['caught']], tw[~tw['caught']]
    print(f"all {len(w)} escalations: median {w['wait'].median():.1f} h after the decision hour, "
          f"{(w['wait'] <= 6).mean():.1%} within 6 h, {(w['wait'] <= 24).mean():.1%} within 24 h; test period: median "
          f"{tw['wait'].median():.1f} h; caught {len(c)}: median {c['wait'].median():.1f} h ({int((c['wait'] <= 6).sum())} "
          f"within 6 h); missed {len(m)}: median {m['wait'].median():.1f} h")


def drop(args):
    import run_escalation as rx
    t = pd.read_csv(args.data)
    y = rx.OUTCOMES[args.outcome]
    d0 = t[(t['landmark'] == args.landmark) & t[y].notna()].reset_index(drop=True)

    def run(skip=None):
        d = d0[d0['symbol'] != skip].reset_index(drop=True) if skip else d0
        r = rx.inference(d, y, args.outcome, args.spec, args.boot)
        c = {x['feature']: x for x in r['coef']}.get(args.feature)
        return None if c is None else (round(c['odds_ratio'], 3), c['p_boot']), r['wald_E'].get('p_boot')

    full, block = run()
    print(f'all assets: odds ratio {full[0]}, bootstrap p {full[1]}; exposure block p {block}')
    sub = d0[d0['has_exposure'] > 0] if args.spec == 'exposed' else d0
    out = [(s,) + run(s) for s in sorted(sub['symbol'].unique())]
    ors = [c[0] for _, c, _ in out if c]
    print(f'dropping one asset at a time: odds ratio {min(ors)} to {max(ors)}, bootstrap p up to '
          f'{max(c[1] for _, c, _ in out if c)}; p >= 0.05 without: {[s for s, c, _ in out if c and c[1] >= 0.05]}')


# ---------------------------------------------------------------- exposure and contagion
def treeseeds(args):
    """Single-seed onset trees, seed by seed: the price trees' test AP and the change in AP that exposure brings on
    hours with exposure (Sections 5 and 8). The published trees average the scores of these seeds."""
    import run_baselines as rb
    from sklearn.metrics import average_precision_score as aps
    d = pd.read_pickle(args.root / 'data' / 'model' / 'dataset.pkl.gz')
    sets = rb.feature_sets(json.loads((args.root / 'data' / 'model' / 'dataset_features.json').read_text()))
    for h in (24, 72):
        y = f'y{h}'
        m = d[d[y].notna()]
        tr, va, te = (m[m['split'] == s].reset_index(drop=True) for s in ('train', 'valid', 'test'))
        yy, exp = te[y].to_numpy(), te['has_exposure'].to_numpy() > 0
        for seed in range(args.n_seeds):
            _, p, ip = rb.fit_hgb(tr, va, te, y, sets['hgb_P'], 300, seed=seed)
            _, pe, _ = rb.fit_hgb(tr, va, te, y, sets['hgb_PE'], 300, seed=seed)
            print(f"{h} h, seed {seed}: price trees AP {aps(yy, p):.4f} ({ip['best_iter']} iterations); exposure changes "
                  f"AP on hours with exposure by {aps(yy[exp], pe[exp]) - aps(yy[exp], p[exp]):+.4f}")


def seedset(args):
    """The published onset trees average seeds 0 .. n-1; refit them on the next n seeds (5 to 9 by default) and compare
    them as compare_preds.py does, with the same asset-block bootstrap (Section 8)."""
    import run_baselines as rb
    from compare_preds import compare
    from sklearn.metrics import average_precision_score as aps
    d = pd.read_pickle(args.root / 'data' / 'model' / 'dataset.pkl.gz')
    sets = rb.feature_sets(json.loads((args.root / 'data' / 'model' / 'dataset_features.json').read_text()))
    seeds = range(args.n_seeds, 2 * args.n_seeds)
    for h in (24, 72):
        y = f'y{h}'
        m = d[d[y].notna()]
        tr, va, te = (m[m['split'] == s].reset_index(drop=True) for s in ('train', 'valid', 'test'))
        base = pd.DataFrame({'symbol': te['symbol'], 'ts': te['ts'], 'y': te[y].astype(int),
                             'has_exposure': te['has_exposure']})
        score = {}
        for name in ('hgb_P', 'hgb_PE', 'hgb_PEd'):
            _, score[name], info = rb.fit_hgb_seeds(tr, va, te, y, sets[name], 300, seeds)
            print(f"{h} h, seeds {seeds.start}-{seeds.stop - 1}: {name} AP {aps(base['y'], score[name]):.4f} "
                  f"(iterations {info['best_iter']})", flush=True)
        for name in ('hgb_PE', 'hgb_PEd'):
            for k, v in compare(base.assign(score=score[name]), base.assign(score=score['hgb_P']), 1000, 0).items():
                print(f"{h} h, {name} - hgb_P [{k}]: {v['diff']:+.5f} ({v['ci'][0]:+.5f} to {v['ci'][1]:+.5f})")


def drift(args):
    d = pd.read_pickle(args.root / 'data' / 'model' / 'dataset.pkl.gz')
    feats = json.loads((args.root / 'data' / 'model' / 'dataset_features.json').read_text())['E']
    rows = []
    for s, g in d[d['has_exposure'] > 0].groupby('symbol'):
        tr, te = g[g['split'] == 'train'], g[g['split'] == 'test']
        if len(tr) < 500 or len(te) < 500:
            continue
        for c in (f for f in feats if f.startswith('log_')):
            a, b = tr[c].dropna(), te[c].dropna()
            if len(a) < 100 or len(b) < 100 or ((a != 0).mean() < 0.5 and (b != 0).mean() < 0.5):
                continue
            rows.append((s, b.median() < a.quantile(0.05) or b.median() > a.quantile(0.95), abs(b.median() - a.median())))
    r = pd.DataFrame(rows, columns=['symbol', 'outside', 'change'])
    print(f"{r['symbol'].nunique()} assets with >= 500 exposed hours in training and in test; the test median of a "
          f"dollar measure lies outside the training 5th-95th percentiles in {int(r['outside'].sum())} of {len(r)} "
          f"cases; median absolute change {r['change'].median():.2f} log units (x{np.exp(r['change'].median()):.1f})")


def seeds(args):
    import run_contagion as rc
    import run_escalation as rx
    pr = read_preds(args.root / 'data' / 'model' / 'contagion_severe_preds.csv.gz')
    d = pr[(pr['window'] == 168) & (pr['protocol'] == 'time')]
    ap = {}
    for m in ('logit_sim', 'logit_fam', 'logit_graph'):
        g = d[d['model'] == m]
        jit = rc.jitter(g['seed_id'], g['cand'])
        g = g.assign(s=g['score'].to_numpy() + jit)
        ap[m] = g.groupby('seed_id').apply(lambda x: rx.ap_score(x['y'].to_numpy(), x['s'].to_numpy())
                                           if x['y'].sum() > 0 else np.nan).dropna()
    for a, b in (('logit_fam', 'logit_sim'), ('logit_graph', 'logit_fam')):
        diff = ap[a] - ap[b]
        moved = diff[diff.abs() > 1e-12].sort_values()
        print(f'{a} - {b}: mean {diff.mean():+.4f} over {len(diff)} seeds; {len(moved)} seeds change; '
              f'without the largest gain {diff.drop(diff.idxmax()).mean():+.4f}')
        print('   ' + ', '.join(f'{k} {v:+.3f}' for k, v in moved.items()))


def _pairs(root):
    d = pd.read_csv(root / 'data' / 'model' / 'contagion_severe.csv.gz')
    return d[d['y168'].notna()].reset_index(drop=True), json.loads((root / 'data' / 'model' / 'contagion_features.json').read_text())


def separation(args):
    import run_escalation as rx
    from sklearn.linear_model import LogisticRegression
    d, g = _pairs(args.root)
    for c in g['GRAPH']:
        nz = d[c].fillna(0) != 0
        print(f'{c:28s} nonzero for {int(nz.sum()):3d} pairs, {int(d.loc[nz, "y168"].sum())} of which followed')
    cols = g['OWN'] + g['SIM'] + g['FAM'] + g['GRAPH']
    X = d[cols].astype(float)
    X = X.clip(*winsor_bounds(X, 0.01, 0.99), axis=1).fillna(X.median()).fillna(0.0)
    X = X.loc[:, X.std() > 1e-9]
    corr = X.corr().abs().to_numpy()
    keep = []
    for i in range(X.shape[1]):
        if all(corr[i, k] < 0.995 for k in keep):
            keep.append(i)
    X = X.iloc[:, keep]
    Xs = ((X - X.mean()) / X.std()).to_numpy()
    y, codes = d['y168'].to_numpy(float), pd.factorize(d['cluster'])[0]
    X1 = np.column_stack([np.ones(len(y)), Xs])
    n, k = X1.shape
    gi = [i + 1 for i, c in enumerate(X.columns) if c in g['GRAPH']]

    def loglik(b):
        z = X1 @ b
        return float(np.sum(y * z - np.logaddexp(0, z)))

    def wald(b):
        p = 1 / (1 + np.exp(-np.clip(X1 @ b, -35, 35)))
        Hm = (X1 * (p * (1 - p))[:, None]).T @ X1 + np.diag(np.r_[0, np.full(k - 1, 1e-4)])
        Hi = np.linalg.inv(Hm)
        U = np.column_stack([np.bincount(codes, weights=X1[:, j] * (y - p)) for j in range(k)])
        G = codes.max() + 1
        V = Hi @ (U.T @ U) @ Hi * G / (G - 1) * (n - 1) / (n - k)
        bb = b[gi]
        return float(bb @ np.linalg.solve(V[np.ix_(gi, gi)], bb))

    beta, _, conv, _ = rx.logit_clustered(Xs, y, d['cluster'].to_numpy())
    print(f'pipeline fit (converged {conv}): log-likelihood {loglik(beta):.4f}, Wald statistic of the links {wald(beta):.1f}')
    for C in (1e2, 1e4, 1e6):
        m = LogisticRegression(C=C, max_iter=50000, tol=1e-12).fit(Xs, y)
        b = np.r_[m.intercept_, m.coef_[0]]
        print(f'refit with a ridge of 1/C, C = {C:g}: log-likelihood {loglik(b):.4f}, Wald statistic {wald(b):.1f}')


def shock(args, reps=4000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    d, g = _pairs(args.root)
    a = d[g['OWN']].astype(float)
    a = a.clip(*winsor_bounds(a, 0.01, 0.99), axis=1).fillna(a.median()).fillna(0.0)
    X = StandardScaler().fit_transform(a)
    p0 = LogisticRegression(C=1, max_iter=5000).fit(X, d['y168']).predict_proba(X)[:, 1]
    linked = (d[g['GRAPH']].fillna(0).gt(0).any(axis=1) & (d['same_family'] == 0)).to_numpy()
    y = d['y168'].to_numpy()
    gap = y[linked].mean() - y[~linked].mean()
    codes = pd.factorize(d['seed_id'])[0]
    S = codes.max() + 1
    O, E = np.bincount(codes, weights=y, minlength=S), np.bincount(codes, weights=p0, minlength=S)
    V = np.bincount(codes, weights=p0 * (1 - p0), minlength=S)
    print(f'observed gap {100 * gap:+.1f} points; dispersion of follower counts across seeds against the own-state '
          f'model {((O - E) ** 2 / V).sum() / (S - 1):.2f} (1 = independent pairs)')
    for var in (0.0, 0.5, 1.0):
        out = []
        for rr, seed in ((1, 1), (2, 2), (3, 3)):
            rng = np.random.default_rng(seed)
            base = np.where(linked, rr * p0, p0)
            low, disp = 0, []
            for _ in range(reps):
                u = rng.gamma(1 / var, var, S) if var > 0 else np.ones(S)
                ys = (rng.uniform(size=len(base)) < np.minimum(base * u[codes], 0.95)).astype(float)
                low += ys[linked].mean() - ys[~linked].mean() <= gap
                if rr == 1:
                    Os = np.bincount(codes, weights=ys, minlength=S)
                    disp.append(((Os - E) ** 2 / V).sum() / (S - 1))
            out.append(f'RR {rr}: {low / reps:.1%}' + (f' (simulated dispersion {np.mean(disp):.2f})' if disp else ''))
        print(f'shock variance {var}: share of replicates at or below the observed gap: ' + ', '.join(out))


def synthetic(args):
    sys.path.insert(0, str(ROOT / 'tests'))
    import run_contagion as rc
    import test_contagion as tc
    with tempfile.TemporaryDirectory() as d:
        table, _ = tc.build(d, tc.synthetic(seed=args.seed, p_link=args.p_link, p_other=args.p_other))
        sys.argv = ['x', '--data', f'{d}/model/contagion_severe.csv.gz', '--out', f'{d}/model', '--bootstrap', '2000',
                    '--inference-bootstrap', '499', '--models', 'random', 'hist', 'follow', 'logit_own', 'logit_sim',
                    'logit_fam', 'logit_graph']
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main()
        res = {r['window']: r for r in json.loads(Path(f'{d}/model/contagion_severe_results.json').read_text())}
    r = res[168]
    for prot in ('time', 'loco'):
        x = r['protocols'][prot]['pairs']['logit_graph - logit_fam']
        print(f"{prot}: links change AP by {x['ap']:+.3f} ({x['ap_ci'][0]:+.3f} to {x['ap_ci'][1]:+.3f})")
    print('link coefficients:', [(c['feature'], round(c['odds_ratio'], 2), c['p_boot']) for c in r['inference']['coef']
                                 if c['block'] == 'GRAPH'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['repeats', 'horizon', 'monitor', 'waits', 'drop', 'drift', 'seeds', 'separation',
                                     'shock', 'synthetic', 'treeseeds', 'seedset', 'chance'])
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--data', default=str(ROOT / 'data' / 'model' / 'escalation.csv.gz'))
    ap.add_argument('--outcome', default='long')
    ap.add_argument('--landmark', type=int, default=6)
    ap.add_argument('--spec', default='exposed')
    ap.add_argument('--feature', default='dlog_family_borrow_against_usd_24h')
    ap.add_argument('--boot', type=int, default=999)
    ap.add_argument('--p-link', type=float, default=0.7)
    ap.add_argument('--p-other', type=float, default=0.04)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-seeds', type=int, default=5, help='treeseeds: seeds 0 .. n-1; seedset: seeds n .. 2n-1')
    args = ap.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
