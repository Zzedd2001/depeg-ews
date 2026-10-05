#!/usr/bin/env python3
"""Pairs for the contagion task (RQ3): after asset s depegs, which other assets depeg next?

One row per (seed episode, candidate asset). Seeds are episodes that passed make_labels' quality checks
and started on or after --start (the exposure graph begins 2024-01-06):
  --seeds severe (default)  episodes that became severe, decided at the hour their held depth first
                            reached -5 % (the second of two consecutive samples), when "s has severely
                            depegged" is known
  --seeds all               every episode, decided when it is known: one hour after its start, when its
                            second sample confirms it, or later if a gap delays that sample (common.known_ts)
Seeds less than 72 h apart form one cluster (they share the market moment and the followers); the
bootstrap in run_contagion.py resamples clusters, and the time split puts a whole cluster on the side
of its first seed (training: decided by 2025-09-30).

Candidates at the decision hour tau: every other labelled asset that has a price in the 24 h before tau
(a deviation carried over a gap of up to 3 h counts), has not collapsed (make_labels' terminal episode started
by tau) and is not inside an open episode (one that started by tau and has not had 24 h back inside the band
by tau, so it cannot start a new one). Like the onset rows inside episodes, these rules use the episodes as
make_labels dates them: an asset whose own episode starts at tau, or started before tau but is confirmed only
after it, is left out. --candidates known instead leaves out only episodes known by tau (common.known_ts), a
sensitivity check in which such assets stay in as candidates that do not follow.
Labels: y72 / y168 = the candidate starts an episode in (tau, tau + 72 h] / (tau, tau + 168 h]; empty when
the window runs past the data. Flagged (suspect) episodes count neither as seeds nor as followers.

Features, all known at tau:
  OWN    the candidate itself: deviation statistics and past episodes (make_dataset.py P group at tau),
         and how often it started an episode within a week after another asset's episode in the past
  SIM    similarity to the seed without the graph: same category, same peg, correlation of hourly
         deviations over the last 30 days, how often the candidate followed this seed's past episodes
  FAM    same token family (wrapper / underlying / PT and other derivatives; config/wrappers.json and
         data/graph/edges_static.csv)
  GRAPH  lending links at the last daily exposure-graph snapshot at or before tau (data/graph/edges.csv.gz).
         Morpho: USD of the candidate's family lent against the seed's family as collateral and the
         reverse, USD that MetaMorpho vaults hold in both families' collateral markets (sum over vaults
         of the smaller side), the share of the candidate's vault funding that comes from vaults exposed
         to the seed (each vault weighted by its seed share), and the number of vaults with >= $10k in both.
         Aave v3 / Spark (shared pools, so links are pool-level upper bounds): USD that both families
         post as collateral in the same pool (sum over pools of the smaller side), and USD of the
         candidate's family borrowed from pools where the seed's family is collateral (sum over pools of
         the smaller of the two). A supply counts as collateral where the reserve has a liquidation
         threshold or an e-mode LTV above zero.
         Zero for pairs in the same family, which FAM covers.

Usage:
  python scripts/make_contagion.py                  # severe seeds -> data/model/contagion_severe.csv.gz
  python scripts/make_contagion.py --seeds all      # every episode -> data/model/contagion_all.csv.gz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_dataset as md  # noqa: E402
from common import ROOT, epoch_seconds, known_ts, label_columns, load_registry, not_suspect, observed_only, to_ts  # noqa: E402,E501

H = 3600
WINDOWS = (72, 168)
CLUSTER_GAP_H = 72
FOLLOW_H = 168                       # "followed" = an episode start within a week after another's start
PRIOR = (0.5, 9.5)                   # Beta prior for follow shares (mean 0.05, close to the weekly base rate)
VAULT_MIN_USD = 1e4
OWN = ['z_now', 'z_min_24h', 'z_min_168h', 'z_std_168h', 'near_frac_24h', 'near_miss_168h', 'hours_since_near_miss',
       'episodes_90d', 'episodes_365d', 'hours_since_episode', 'peg_eth', 'und_z_now', 'missing_frac_24h',
       'follow_rate']
SIM = ['same_category', 'same_peg', 'dev_corr_30d', 'codepeg_share']
FAM = ['same_family']
GRAPH_MORPHO = ['log_lend_cand_vs_seed', 'log_lend_seed_vs_cand', 'log_vault_overlap', 'vault_share', 'shared_vaults']
GRAPH_POOL = ['log_pool_overlap', 'log_pool_lend_cand_vs_seed']
GRAPH = GRAPH_MORPHO + GRAPH_POOL
EDGE_COLS = ['t', 'src', 'dst', 'etype', 'usd', 'liq_threshold', 'emode_ltv']


# ---------------------------------------------------------------- families
def families(static_edges, nodes):
    """Union-find over wrapper / derivative edges (undirected). Returns node -> family root."""
    parent = {n: n for n in nodes}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in static_edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {n: find(n) for n in list(parent)}


# ---------------------------------------------------------------- seeds
def severe_hour(values, index, sev_level):
    """First hour at which two consecutive known samples are both at or beyond -sev_level."""
    ok = ~np.isnan(values)
    v, ix = values[ok], index[ok]
    hit = np.flatnonzero(np.maximum(v[:-1], v[1:]) <= -sev_level)
    return int(ix[hit[0] + 1]) if len(hit) else None


def seed_table(ep, dev, kind, start_ts, sev_level):
    """dev: symbol -> deviation at observed hours (held depth is measured on samples, not carried-forward hours);
    ep may carry tk, the hour each episode is known (common.known_ts; default start + 1 h)."""
    rows = []
    for e in ep[ep['t0'] >= start_ts].itertuples():
        tk = int(getattr(e, 'tk', e.t0 + H))
        if kind == 'severe':
            if e.severity not in ('major', 'collapse'):
                continue
            d = dev[e.symbol].loc[e.t0:e.t1]
            tau = severe_hour(d.to_numpy(dtype=float), d.index.to_numpy(), sev_level)
            tau = tk if tau is None else max(tau, tk)
        else:
            tau = tk                        # one hour after the start, or later if a gap delays confirmation
        rows.append({'seed_id': f'{e.symbol}@{e.start}', 'seed': e.symbol, 'seed_t0': e.t0, 'seed_severity': e.severity,
                     'tau': int(tau)})
    if not rows:
        sys.exit(f'no {kind} seeds on or after the start date')
    seeds = pd.DataFrame(rows).sort_values(['tau', 'seed'], kind='stable').reset_index(drop=True)
    gap = seeds['tau'].diff().fillna(np.inf) > CLUSTER_GAP_H * H
    seeds['cluster'] = np.cumsum(gap.to_numpy()) - 1
    return seeds


# ---------------------------------------------------------------- candidates and labels
def candidate_rows(seeds, ep, dev, symbols, data_end, known_only=False):
    """ep: clean episodes with t0, t1, tk, terminal; dev: symbol -> deviation Series (sorted int index).
    known_only: episodes and collapses count from the hour they are known (tk) rather than from their start."""
    t_ep = 'tk' if known_only else 't0'
    by_sym = {s: g.sort_values('t0') for s, g in ep.groupby('symbol')}
    death = ep[ep['terminal']].groupby('symbol')[t_ep].min().to_dict()
    known = {s: d.dropna().index.to_numpy() for s, d in dev.items()}
    rows = []
    for sd in seeds.itertuples():
        tau = sd.tau
        for j in symbols:
            if j == sd.seed or (j in death and death[j] <= tau):
                continue
            k = known[j]
            lo = np.searchsorted(k, tau - 24 * H, side='left')
            if lo >= len(k) or k[lo] > tau:
                continue                                          # no price in the 24 h before tau
            g = by_sym.get(j)
            if g is not None and ((g[t_ep] <= tau) & (g['t1'] > tau - 24 * H)).any():
                continue                                          # inside an open episode
            row = {'seed_id': sd.seed_id, 'seed': sd.seed, 'seed_severity': sd.seed_severity, 'seed_t0': sd.seed_t0,
                   'tau': tau, 'cluster': sd.cluster, 'cand': j}
            starts = g['t0'].to_numpy() if g is not None else np.array([], dtype='int64')
            for w in WINDOWS:
                row[f'y{w}'] = np.nan if tau + w * H > data_end else float(((starts > tau) & (starts <= tau + w * H)).any())
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- history-based similarity
def follow_features(pairs, ep, symbols):
    """follow_rate: share of other assets' episodes (with their week fully observed by tau) that the
    candidate followed within a week; codepeg_share: the same over the seed's own episodes."""
    e = ep.sort_values('t0', kind='stable').reset_index(drop=True)
    t0 = e['t0'].to_numpy()
    sym = e['symbol'].to_numpy()
    starts = {s: np.sort(g['t0'].to_numpy()) for s, g in e.groupby('symbol')}
    F = np.zeros((len(e), len(symbols)))                          # F[i, j]: asset j started within a week after episode i
    for jj, j in enumerate(symbols):
        st = starts.get(j, np.array([], dtype='int64'))
        if len(st):
            nxt = np.searchsorted(st, t0, side='right')
            ok = nxt < len(st)
            F[ok, jj] = (st[np.minimum(nxt, len(st) - 1)][ok] <= t0[ok] + FOLLOW_H * H).astype(float)
    other = sym[:, None] != np.array(symbols)[None, :]
    cumF = np.vstack([np.zeros(len(symbols)), np.cumsum(F * other, axis=0)])
    cumN = np.vstack([np.zeros(len(symbols)), np.cumsum(other, axis=0)])
    col = {j: i for i, j in enumerate(symbols)}
    n_done = np.searchsorted(t0, pairs['tau'].to_numpy() - (FOLLOW_H + 1) * H, side='right')   # weeks closed by tau
    jc = pairs['cand'].map(col).to_numpy()
    k, n = cumF[n_done, jc], cumN[n_done, jc]
    out = pd.DataFrame(index=pairs.index)
    out['follow_rate'] = (k + PRIOR[0]) / (n + sum(PRIOR))
    # the seed's own past episodes
    own_idx = {s: np.flatnonzero(sym == s) for s in np.unique(sym)}
    ks, ns = np.zeros(len(pairs)), np.zeros(len(pairs))
    for i, (s, tau, j) in enumerate(zip(pairs['seed'], pairs['tau'], jc)):
        idx = own_idx.get(s, np.array([], dtype=int))
        idx = idx[t0[idx] <= tau - (FOLLOW_H + 1) * H]
        ns[i], ks[i] = len(idx), F[idx, j].sum()
    out['codepeg_share'] = (ks + PRIOR[0]) / (ns + sum(PRIOR))
    return out


def corr_features(pairs, z, min_obs=240):
    """Correlation of hourly threshold-unit deviations of seed and candidate over the 720 h up to tau."""
    out = np.full(len(pairs), np.nan)
    cols = {c: i for i, c in enumerate(z.columns)}
    zi = z.index.to_numpy()
    zv = z.to_numpy(dtype=float)
    for (s, tau), g in pairs.groupby(['seed', 'tau'], sort=False):
        lo, hi = np.searchsorted(zi, tau - 719 * H, side='left'), np.searchsorted(zi, tau, side='right')
        x = zv[lo:hi, cols[s]]
        for i, j in zip(g.index, g['cand']):
            y = zv[lo:hi, cols[j]]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= min_obs and x[ok].std() > 0 and y[ok].std() > 0:
                out[pairs.index.get_loc(i)] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return out


# ---------------------------------------------------------------- exposure-graph links
def _zero_if_missing(df, col):
    return df[col].fillna(0.0) if col in df else pd.Series(0.0, index=df.index)


class GraphLinks:
    """Lending and vault links between token families at daily snapshots of data/graph/edges.csv.gz:
    Morpho markets and MetaMorpho vaults, and Aave v3 / Spark pools (etypes pool_supply / pool_borrow;
    without the liq_threshold / emode_ltv columns no supply counts as collateral)."""

    def __init__(self, edges, static, symbols):
        tok = [f'tok:{s}' for s in symbols]
        nodes = set(tok) | set(static['src']) | set(static['dst']) | set(edges.loc[edges['etype'] == 'collateral', 'src'])
        self.fam = families(zip(static['src'], static['dst']), nodes)
        col = edges[edges['etype'] == 'collateral'].groupby('dst')['src'].agg(lambda s: s.mode().iat[0])
        loan = edges[edges['etype'] == 'borrow'].groupby('src')['dst'].agg(lambda s: s.mode().iat[0])
        fam = lambda n: self.fam.get(n, n)
        self.col_fam = col.map(fam).to_dict()              # market -> family of its collateral
        self.loan_fam = loan.map(fam).to_dict()            # market -> family of its loan asset
        self.borrow = edges.loc[edges['etype'] == 'borrow', ['t', 'src', 'usd']]
        self.alloc = edges.loc[edges['etype'] == 'allocation', ['t', 'src', 'dst', 'usd']]
        ps = edges[edges['etype'] == 'pool_supply']
        is_col = (_zero_if_missing(ps, 'liq_threshold') > 0) | (_zero_if_missing(ps, 'emode_ltv') > 0)
        ps = ps[is_col]
        self.pool_col = pd.DataFrame({'t': ps['t'], 'pool': ps['dst'], 'fam': ps['src'].map(fam), 'usd': ps['usd']})
        pb = edges[edges['etype'] == 'pool_borrow']
        self.pool_debt = pd.DataFrame({'t': pb['t'], 'pool': pb['src'], 'fam': pb['dst'].map(fam), 'usd': pb['usd']})
        self.times = np.sort(edges['t'].unique())
        self.sym_fam = {s: fam(f'tok:{s}') for s in symbols}
        self.cache = {}

    def snapshot(self, tau):
        i = np.searchsorted(self.times, tau, side='right') - 1
        return None if i < 0 else int(self.times[i])

    def tables(self, t):
        if t in self.cache:
            return self.cache[t]
        b = self.borrow[self.borrow['t'] == t]
        lend = pd.DataFrame({'col': b['src'].map(self.col_fam), 'loan': b['src'].map(self.loan_fam), 'usd': b['usd']})
        lend = lend.dropna(subset=['col', 'loan']).groupby(['col', 'loan'])['usd'].sum().to_dict()
        a = self.alloc[self.alloc['t'] == t]
        total = a.groupby('src')['usd'].sum()
        a = pd.DataFrame({'vault': a['src'], 'fam': a['dst'].map(self.col_fam), 'usd': a['usd']}).dropna(subset=['fam'])
        expo = a.groupby(['vault', 'fam'])['usd'].sum().unstack(fill_value=0.0)
        per_pool = lambda x: x[x['t'] == t].groupby(['pool', 'fam'])['usd'].sum().unstack(fill_value=0.0)
        pc, pdebt = per_pool(self.pool_col), per_pool(self.pool_debt)          # pool x family: collateral, debt
        pools = pc.index.union(pdebt.index)
        self.cache[t] = (lend, expo, total.reindex(expo.index).fillna(0.0),
                         pc.reindex(pools, fill_value=0.0), pdebt.reindex(pools, fill_value=0.0))
        return self.cache[t]

    def features(self, seed, cand, tau):
        fs, fc = self.sym_fam[seed], self.sym_fam[cand]
        out = dict.fromkeys(GRAPH, 0.0)
        out['same_family'] = float(fs == fc)
        t = self.snapshot(tau)
        if t is None or fs == fc:
            return out
        lend, expo, total, pc, pdebt = self.tables(t)
        out['log_lend_cand_vs_seed'] = float(np.log1p(lend.get((fs, fc), 0.0)))
        out['log_lend_seed_vs_cand'] = float(np.log1p(lend.get((fc, fs), 0.0)))
        if fs in expo.columns and fc in expo.columns:
            a_s, a_c = expo[fs].to_numpy(), expo[fc].to_numpy()
            tot = total.to_numpy()
            out['log_vault_overlap'] = float(np.log1p(np.minimum(a_s, a_c).sum()))
            if a_c.sum() > 0:
                out['vault_share'] = float((a_c * np.divide(a_s, tot, out=np.zeros_like(a_s), where=tot > 0)).sum() / a_c.sum())
            out['shared_vaults'] = float(((a_s >= VAULT_MIN_USD) & (a_c >= VAULT_MIN_USD)).sum())
        if fs in pc.columns:
            c_s = pc[fs].to_numpy()
            if fc in pc.columns:
                out['log_pool_overlap'] = float(np.log1p(np.minimum(c_s, pc[fc].to_numpy()).sum()))
            if fc in pdebt.columns:
                out['log_pool_lend_cand_vs_seed'] = float(np.log1p(np.minimum(c_s, pdebt[fc].to_numpy()).sum()))
        return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    for name, default in (('registry', 'config/assets.json'), ('wrappers', 'config/wrappers.json'),
                          ('rules', 'config/label_rules.json'), ('labels', 'data/labels/labels_hourly.csv.gz'),
                          ('episodes', 'data/labels/episodes.csv'), ('prices', 'data/prices_llama_hourly.csv.gz'),
                          ('features', 'data/graph_hourly/token_features.csv.gz'), ('graph', 'data/graph')):
        ap.add_argument(f'--{name}', default=str(ROOT / default))
    ap.add_argument('--seeds', choices=['severe', 'all'], default='severe')
    ap.add_argument('--candidates', choices=['dated', 'known'], default='dated',
                    help='leave out assets inside episodes as dated by make_labels (default), or only inside episodes '
                         'already known at the decision hour (sensitivity check)')
    ap.add_argument('--start', default='2024-01-07', help='first seed start (the daily exposure graph begins 2024-01-06)')
    ap.add_argument('--out', help='default: data/model/contagion_<seeds>.csv.gz')
    args = ap.parse_args()
    static_path = Path(args.graph) / 'edges_static.csv'
    if not static_path.exists():                       # without it every token would silently be its own family
        raise SystemExit(f'{static_path} is missing: build_graph.py writes the token families (wrappers and '
                         'derivatives) there; rebuild it or copy it with the rest of data/graph')

    rules = json.loads(Path(args.rules).read_text())
    _, assets = load_registry(args.registry)
    meta = {a['symbol']: a for a in assets}
    sev_level = rules['severity_tiers'][0][0]
    lab = pd.read_csv(args.labels, usecols=label_columns(args.labels))
    lab['ts'] = epoch_seconds(lab['hour'])
    lab['dev_obs'] = observed_only(lab)
    symbols = sorted(lab['symbol'].unique())
    dev = {s: g.set_index('ts')['dev'].sort_index() for s, g in lab.groupby('symbol')}
    dev_obs = {s: g.set_index('ts')['dev_obs'].sort_index() for s, g in lab.groupby('symbol')}
    thr = {s: rules['threshold'][meta[s]['peg']] for s in symbols}
    index = np.arange(lab['ts'].min(), lab['ts'].max() + H, H, dtype='int64')
    z = lab.pivot(index='ts', columns='symbol', values='dev').reindex(index=index, columns=symbols).clip(-1, 1) / pd.Series(thr)
    data_end = int(lab['ts'].max())
    del lab

    ep = not_suspect(pd.read_csv(args.episodes))
    ep = ep[ep['symbol'].isin(symbols)].copy()
    ep['t0'], ep['t1'] = epoch_seconds(ep['start']), epoch_seconds(ep['end'])
    ep['tk'] = np.asarray(known_ts(ep), dtype='int64')
    ep['terminal'] = ep['terminal'].astype(str).str.strip().str.lower().isin(['true', '1', '1.0']) if 'terminal' in ep else False
    seeds = seed_table(ep, dev_obs, args.seeds, to_ts(args.start), sev_level)
    pairs = candidate_rows(seeds, ep, dev, symbols, data_end, known_only=args.candidates == 'known')

    # SIM
    pairs['same_category'] = (pairs['seed'].map(lambda s: meta[s]['category']) == pairs['cand'].map(lambda s: meta[s]['category'])).astype(float)
    pairs['same_peg'] = (pairs['seed'].map(lambda s: meta[s]['peg']) == pairs['cand'].map(lambda s: meta[s]['peg'])).astype(float)
    pairs['dev_corr_30d'] = corr_features(pairs, z)
    del z
    pairs = pd.concat([pairs, follow_features(pairs, ep, symbols)], axis=1)

    # FAM + GRAPH
    g = Path(args.graph)
    have = pd.read_csv(g / 'edges.csv.gz', nrows=0).columns
    edges = pd.read_csv(g / 'edges.csv.gz', usecols=[c for c in EDGE_COLS if c in have])
    static = pd.read_csv(g / 'edges_static.csv')
    links = GraphLinks(edges, static, symbols)
    del edges
    gf = pd.DataFrame([links.features(s, c, t) for s, c, t in zip(pairs['seed'], pairs['cand'], pairs['tau'])], index=pairs.index)
    pairs = pd.concat([pairs, gf[FAM + GRAPH]], axis=1)

    # OWN: the candidate's hourly features at tau
    pairs['row_id'] = np.arange(len(pairs))
    feats, _ = md.build(args, rows=pairs[['tau', 'cand', 'row_id']].rename(columns={'tau': 'ts', 'cand': 'symbol'}))
    own = [c for c in OWN if c != 'follow_rate']
    pairs = pairs.merge(feats[['row_id'] + own], on='row_id', how='inner').drop(columns='row_id')
    tr, va = to_ts(rules['split']['train_end'][:16]), to_ts(rules['split']['valid_end'][:16])
    first = pairs.groupby('cluster')['tau'].transform('min')               # a cluster stays on one side of the split
    pairs['split'] = np.where(first <= va, 'train', 'test')                # (validation months join training here)
    pairs['seed_time'] = pd.to_datetime(pairs['tau'], unit='s', utc=True).dt.strftime('%Y-%m-%dT%H:00Z')
    del tr

    out = Path(args.out or ROOT / 'data' / 'model' / f'contagion_{args.seeds}.csv.gz')
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out, index=False, float_format='%.6g')
    (out.parent / 'contagion_features.json').write_text(json.dumps(
        {'OWN': OWN, 'SIM': SIM, 'FAM': FAM, 'GRAPH': GRAPH, 'GRAPH_MORPHO': GRAPH_MORPHO, 'GRAPH_POOL': GRAPH_POOL}, indent=1))
    s = pairs.groupby('split').agg(seeds=('seed_id', 'nunique'), clusters=('cluster', 'nunique'), pairs=('cand', 'size'),
                                   y72=('y72', 'sum'), y168=('y168', 'sum'))
    print(f'wrote {out}: {len(pairs)} pairs from {pairs["seed_id"].nunique()} {args.seeds} seeds; '
          f'{int(pairs["same_family"].sum())} same-family pairs')
    for name, cols in (('any', GRAPH), ('a Morpho', GRAPH_MORPHO), ('an Aave / Spark pool', GRAPH_POOL)):
        linked = pairs[cols].gt(0).any(axis=1)
        print(f'  pairs with {name} link: {int(linked.sum())} ({int(pairs.loc[linked, "y168"].sum())} followed within a week)')
    print(s.astype(int).to_string())


if __name__ == '__main__':
    main()
