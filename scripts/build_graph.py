#!/usr/bin/env python3
"""Exposure-graph snapshots and per-token exposure features, without look-ahead.

Inputs (each optional; a missing source is skipped with a note):
  config/assets.json, config/wrappers.json
  data/morpho/markets.csv, vaults.csv, market_history_{day,hour}.csv.gz, vault_allocation_{day,hour}.csv.gz
  data/lending/aave_reserves.csv, aave_emode.csv
  data/prices_llama_hourly.csv.gz   market prices: token-unit values -> USD, and the Aave oracle gap

Timing rule: an observation is used only once it is complete. A Morpho point stamped x (DAY or
HOUR) is used from x + 1 h: the DAY point equals the HOUR point at 00:00, i.e. the state at the
start of the day (checked on 35,185 market-days, 2026-10-03), so it is not a daily average or a
day-end value. An Aave read is the block state at the grid time. A value older than
--max-stale-hours at the snapshot time counts as missing.

Usage:
  python scripts/build_graph.py                          # daily snapshots 2023-01-01 .. 2026-09-30
  python scripts/build_graph.py --step-hours 6
  python scripts/build_graph.py --step-hours 1 --no-edges --out data/graph_hourly   # hourly features only

Outputs (data/graph/):
  nodes.csv              node_id, ntype, symbol, address, category, in_scope, lltv, oracle_class, asset
                         (node ids: tok:<symbol or address>, mm:<market id, first 12 hex>,
                          mv:<vault address, first 12 hex>, pool:<name>; full ids in 'address')
  edges_static.csv       wrapper (config/wrappers.json) and derivative (PT-/YT-/LP-/SY- symbol) edges
  edges.csv.gz           t, time, src, dst, etype, usd, tokens, lltv, ltv, liq_threshold, emode_ltv,
                         oracle_class, frozen; etype: collateral, supply, borrow, allocation,
                         vault_asset, pool_supply, pool_borrow (grouped by etype, sorted by time)
  token_features.csv.gz  symbol, t, time, one column per feature in FEATURES (empty = no data)
  build_report.md        coverage of each source and a feature summary
"""
import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_registry, to_ts  # noqa: E402

LAG = {'day': 3600, 'hour': 3600}      # seconds after its timestamp that a Morpho point may be used (see above)
ZERO = '0x' + '0' * 40
BLIND = {'fixed', 'vault_rate', 'rate_feed'}      # oracle classes that do not see a market depeg of the collateral
DERIV_PREFIX = {'PT', 'YT', 'LP', 'SY'}
EDGE_COLS = ['t', 'time', 'src', 'dst', 'etype', 'usd', 'tokens', 'lltv', 'ltv', 'liq_threshold', 'emode_ltv',
             'oracle_class', 'frozen', 'flag']
FEATURES = {
    'mm_n_markets': 'Morpho markets with this token as collateral and >= min-edge-usd of it posted',
    'mm_collateral_usd': 'USD of the token posted as collateral on Morpho',
    'mm_borrow_against_usd': 'USD borrowed in Morpho markets that take the token as collateral',
    'mm_lltv_wavg': 'collateral-weighted liquidation LTV of those markets (this and the two shares: empty below min-edge-usd posted)',
    'mm_blind_oracle_share': 'share of that collateral in markets whose oracle cannot see a market depeg of it (fixed / vault rate / rate feed)',
    'mm_custom_oracle_share': 'share of that collateral in markets with a non-standard oracle',
    'mm_pegged_loop_usd': 'USD borrowed against the token in an asset with the same peg (USD or ETH): loops that unwind in a depeg',
    'mm_supply_usd': 'USD of the token supplied to Morpho markets as the loan asset',
    'mm_borrowed_usd': 'USD of the token borrowed from Morpho markets',
    'vault_exposure_usd': 'USD that MetaMorpho vaults allocate to markets taking the token as collateral',
    'vault_n': 'vaults allocating >= min-edge-usd to such markets',
    'aave_supply_usd': 'USD of the token supplied to Aave v3 / Spark pools (pool oracle price)',
    'aave_collateral_usd': 'part of that supply in pools where the token counts as collateral (liquidation threshold > 0)',
    'aave_borrow_usd': 'USD of the token borrowed from Aave v3 / Spark pools',
    'aave_ltv_max': 'highest LTV available for the token across pools, e-mode included',
    'aave_frozen': '1 if the token is frozen or paused in any pool that lists it',
    'aave_supply_cap_use': 'highest supply / supply-cap ratio across pools',
    'aave_oracle_gap': 'pool oracle price / market price - 1 (supply-weighted); negative = pool values it below market',
    'family_collateral_usd': 'Morpho + Aave collateral USD of the token and of everything wrapping or derived from it',
    'family_vault_exposure_usd': 'vault exposure to the token and to everything wrapping or derived from it',
    'family_borrow_against_usd': 'Morpho borrowing against the token family plus Aave borrowing of family members',
}


def read_csv(path, **kw):
    p = Path(path)
    if not p.exists() and p.suffix != '.gz' and Path(str(p) + '.gz').exists():
        p = Path(str(p) + '.gz')
    return pd.read_csv(p, **kw) if p.exists() else None


MARKET_FIELDS = ['supplyAssetsUsd', 'borrowAssetsUsd', 'collateralAssetsUsd', 'collateralAssets']
VAULT_FIELDS = ['supplyAssetsUsd', 'totalAssetsUsd', 'supplyAssets', 'totalAssets']


def read_obs(path, cats, lag, prio, blank=None):
    """A Morpho history file with its id columns read as categoricals over fixed category lists: compact
    (the hourly files have millions of rows), and parts with equal categories concatenate as categoricals.
    Rows whose ids are not in the lists are dropped; `blank` names a column whose empty cells mean ''
    (vault totals have no market)."""
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={**{c: 'category' for c in cats}, 'ts': 'int64', 'value': 'float64'})
    if not len(df):
        return None
    empty = df[blank].isna() if blank else None
    for c, values in cats.items():
        df[c] = df[c].cat.set_categories(list(dict.fromkeys(values)))
    if blank:
        df.loc[empty, blank] = ''
    df = df.dropna(subset=list(cats)).reset_index(drop=True)
    df['avail_ts'] = df['ts'] + lag
    df['prio'] = np.int8(prio)
    return df


def decat(s):
    """Categorical -> object column whose cells share one Python string per category (cheap in memory)."""
    if not isinstance(s.dtype, pd.CategoricalDtype):
        return s
    cats = np.asarray(list(s.cat.categories), dtype=object)
    codes = s.cat.codes.to_numpy()
    out = cats.take(np.where(codes < 0, 0, codes))
    out[codes < 0] = np.nan
    return pd.Series(out, index=s.index, dtype=object)


def asof(obs, by, values, grid, max_stale):
    """Value of every key at every grid time t: the latest observation with avail_ts <= t, provided
    t - avail_ts <= max_stale. Only (t, key) pairs that have a value are returned: ['t'] + by + values."""
    if obs is None or not len(obs):
        return pd.DataFrame(columns=['t'] + by + values)
    o = obs.sort_values(by + ['avail_ts'], kind='stable').reset_index(drop=True)
    a = o['avail_ts'].to_numpy(dtype='int64')
    gid = o.groupby(by, sort=False, dropna=False, observed=True).ngroup().to_numpy()
    far = np.iinfo('int64').max // 2
    nxt = np.append(a[1:], far)
    nxt[np.append(gid[1:] != gid[:-1], True)] = far          # last row of each key: no successor
    end = np.minimum(nxt, a + int(max_stale) + 1)              # row i holds for t in [a_i, end_i)
    grid = np.asarray(grid, dtype='int64')
    i0, i1 = np.searchsorted(grid, a, 'left'), np.searchsorted(grid, end, 'left')
    n = np.maximum(i1 - i0, 0)
    rows = np.repeat(np.arange(len(o)), n)
    pos = np.repeat(i0, n) + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))
    out = o.iloc[rows][by + values].reset_index(drop=True)
    out.insert(0, 't', grid[pos])
    return out


def wide_by_market(a, grid):
    """Long as-of rows (t, market_id, field, value; categorical ids) -> one row per (t, market_id) with a
    column per field, sorted by t and market id. Done on integer codes: the hourly build has tens of
    millions of long rows, and pandas' pivot on them needs several GB."""
    mcats = np.asarray(list(a['market_id'].cat.categories), dtype=object)
    fcats = [str(c) for c in a['field'].cat.categories]
    mcode = a['market_id'].cat.codes.to_numpy().astype(np.int64)
    fcode = a['field'].cat.codes.to_numpy()
    tpos = np.searchsorted(grid, a['t'].to_numpy())
    key = tpos.astype(np.int64) * len(mcats) + mcode
    uniq, inv = np.unique(key, return_inverse=True)
    del key
    wide = np.full((len(uniq), len(fcats)), np.nan)
    wide[inv, fcode] = a['value'].to_numpy(dtype='float64')
    del inv
    out = pd.DataFrame({'t': grid[uniq // len(mcats)], 'market_id': mcats.take(uniq % len(mcats))})
    for j, f in enumerate(fcats):
        if not np.isnan(wide[:, j]).all():
            out[f] = wide[:, j]
    return out.sort_values(['t', 'market_id'], kind='stable').reset_index(drop=True)


def price_lookup(prices, symbols, times, tolerance=6 * 3600):
    """Market price (USD) of each symbol at or before each time; NaN if none within tolerance."""
    if prices is None or not len(symbols):
        return np.full(len(symbols), np.nan)
    q = pd.DataFrame({'symbol': pd.Series(symbols, dtype=object).astype(str).to_numpy(),
                      'ts': np.asarray(times, dtype='int64'), 'i': np.arange(len(symbols))}).sort_values('ts', kind='stable')
    out = pd.merge_asof(q, prices, on='ts', by='symbol', direction='backward', tolerance=tolerance)
    return out.sort_values('i')['price'].to_numpy()


def emode_ltv(ar, em):
    """Highest e-mode LTV each reserve row can use (0 if none), from the e-mode rows of the same pool and time.
    Aave 3.2+: membership from the category's collateral bitmap (bit = reserve id); before: the reserve's own
    e-mode category (config bits 168-175)."""
    em = em[em['ltv'] > 0].copy()
    if not len(em):
        return np.zeros(len(ar))
    em['cbm'] = em['collateral_bitmap'].map(lambda x: str(int(x, 16)) if isinstance(x, str) and x.startswith('0x') else '0')
    em['part'] = em['category'].astype(int).astype(str) + ':' + em['ltv'].astype(str) + ':' + em['cbm'] + ':' + em['mode'].astype(str)
    sig = em.sort_values(['pool', 'ts', 'category']).groupby(['pool', 'ts'])['part'].agg('|'.join).rename('sig').reset_index()
    x = ar[['pool', 'ts']].copy()
    x['rid'] = ar['reserve_id'].fillna(-1).astype(int) if 'reserve_id' in ar else -1
    x['leg'] = ar['emode_legacy'].fillna(0).astype(int) if 'emode_legacy' in ar else 0
    x = x.merge(sig, on=['pool', 'ts'], how='left')
    best = {}
    for s, rid, leg in x[['sig', 'rid', 'leg']].dropna().drop_duplicates().itertuples(index=False):
        b = 0.0
        for part in s.split('|'):
            cat, ltv, cbm, mode = part.split(':')
            member = ((int(cbm) >> rid) & 1) if mode == 'bitmap' and rid >= 0 else int(leg == int(cat))
            if member:
                b = max(b, float(ltv))
        best[(s, rid, leg)] = b
    return np.array([best.get((s, r, l), 0.0) if isinstance(s, str) else 0.0
                     for s, r, l in zip(x['sig'], x['rid'], x['leg'])])


FLAG_COLS = ['market_id', 'rule', 'first', 'last', 'snapshots', 'max_raw_borrow', 'max_kept_borrow']


def clean_morpho(mm, va, min_debt=1e5, hold_hours=48, unverifiable_days=2):
    """Two data-quality rules on the snapshot frames. Each decision at time t uses snapshots up to t only.

    bad_debt_freeze: at every snapshot of the last >= 48 h the collateral was worth less than half the
      debt (debt > $100k). Such a market sits at 100 % utilisation and its debt keeps growing on paper at
      the top interest rate (sdeUSD/USDC went from $8.7M to $7.5B after deUSD collapsed); nobody can be
      paid that. Supply and borrow are held at their values at the first insolvent snapshot, and so are
      vault allocations to the market (the vault's total loses the same excess). A snapshot with no
      collateral value (no price) keeps the market's last known state, so a dead token's missing price
      does not end the freeze.
    unverifiable: the collateral has never had a price, and supply equalled borrow (>= 99.9 % utilisation)
      at >= 90 % of the snapshots with > $1M supplied, for >= 2 days: a circular market whose size cannot
      be checked (BONDUSD/USR reached $12.3B). Its USD values are left out.
    Returns (mm, va, flags); mm and va gain a 'flag' column."""
    mm = mm.sort_values(['market_id', 't'], kind='stable').reset_index(drop=True)
    g, t = mm['market_id'], mm['t']
    col, bor, sup = mm['collateralAssetsUsd'], mm['borrowAssetsUsd'].copy(), mm['supplyAssetsUsd'].copy()
    same = g.eq(g.shift())
    # bad_debt_freeze
    state = pd.Series(np.where(col.isna() | bor.isna(), np.nan, (col < 0.5 * bor).astype(float)), index=mm.index)
    ins = (state.groupby(g).ffill() == 1) & (bor > min_debt)      # unknown collateral value: last known state
    run = (ins & ~(ins.shift(fill_value=False) & same)).cumsum().where(ins)
    t0 = t.groupby(run).transform('min')
    frozen = ins & (t - t0 >= hold_hours * 3600)
    mm.loc[frozen, 'supplyAssetsUsd'] = np.fmin(sup[frozen], sup.groupby(run).transform('first')[frozen])
    mm.loc[frozen, 'borrowAssetsUsd'] = np.fmin(bor[frozen], bor.groupby(run).transform('first')[frozen])
    mm['flag'] = np.where(frozen, 'bad_debt_freeze', '')
    mm['t0'] = t0.where(frozen)
    # unverifiable
    big = sup > 1e6
    c_n = big.astype(int).groupby(g).cumsum()
    c_full = (big & (bor >= 0.999 * sup)).astype(int).groupby(g).cumsum()
    c_priced = (col > 0).astype(int).groupby(g).cumsum()
    first_big = t.where(big).groupby(g).cummin().groupby(g).ffill()
    unver = (t - first_big >= unverifiable_days * 86400) & (c_priced == 0) & (c_n > 0) & (c_full >= 0.9 * c_n)
    mm.loc[unver, ['supplyAssetsUsd', 'borrowAssetsUsd', 'collateralAssetsUsd']] = np.nan
    mm.loc[unver, 'flag'] = 'unverifiable'
    fl = pd.DataFrame({'market_id': g, 't': t, 'rule': mm['flag'], 'raw': bor, 'kept': mm['borrowAssetsUsd']})[mm['flag'] != '']
    flags = fl.groupby(['market_id', 'rule']).agg(first=('t', 'min'), last=('t', 'max'), snapshots=('t', 'size'),
                                                   max_raw_borrow=('raw', 'max'), max_kept_borrow=('kept', 'max')).reset_index()
    if len(va):
        va = va.reset_index(drop=True)
        va['flag'] = ''
        hit = mm.loc[mm['flag'] != '', ['market_id', 't', 'flag', 't0']]
        x = va[['vault', 'market_id', 't', 'value']].reset_index().merge(hit, on=['market_id', 't'], how='inner')
        if len(x):
            base = va[['vault', 'market_id', 't', 'value']].rename(columns={'t': 't0', 'value': 'v0'})
            x = x.merge(base, on=['vault', 'market_id', 't0'], how='left')
            kept = np.where(x['flag'] == 'unverifiable', 0.0, np.fmin(x['value'], x['v0']))
            excess = (x['value'] - kept).groupby([x['vault'], x['t']]).sum()
            va.loc[x['index'], 'value'] = kept
            va.loc[x['index'], 'flag'] = x['flag'].to_numpy()
            tot = va.index[va['market_id'] == '']
            ix = pd.MultiIndex.from_arrays([va.loc[tot, 'vault'], va.loc[tot, 't']])
            va.loc[tot, 'value'] = va.loc[tot, 'value'].to_numpy() - excess.reindex(ix).fillna(0.0).to_numpy()
        va = va[va['value'] > 0].reset_index(drop=True)
    return mm.drop(columns='t0'), va, flags.reindex(columns=FLAG_COLS)


class EdgeWriter:
    def __init__(self, path, min_usd):
        self.fh = gzip.open(path, 'wt', newline='')
        self.fh.write(','.join(EDGE_COLS) + '\n')
        self.min_usd, self.n_all, self.counts = min_usd, 0, {}

    def write(self, df):
        df = df.reindex(columns=EDGE_COLS)
        df = df[df['usd'].notna() | df['tokens'].notna()]
        self.n_all += len(df)
        df = df[(df['usd'] >= self.min_usd) | (df['usd'].isna() & (df['tokens'].fillna(0) > 0))].sort_values('t', kind='stable')
        if not len(df):
            return
        df = df.assign(time=pd.to_datetime(df['t'], unit='s', utc=True).dt.strftime('%Y-%m-%dT%H:00Z'))
        df.to_csv(self.fh, header=False, index=False)
        for k, v in df['etype'].value_counts().items():
            self.counts[k] = self.counts.get(k, 0) + int(v)

    def close(self):
        self.fh.close()


class NoEdges(EdgeWriter):
    """--no-edges: count what would be written, write nothing (for fine-grained feature tables)."""

    def __init__(self, min_usd):
        self.min_usd, self.n_all, self.counts = min_usd, 0, {}

    def write(self, df):
        df = df.reindex(columns=EDGE_COLS)
        df = df[df['usd'].notna() | df['tokens'].notna()]
        self.n_all += len(df)
        kept = df[(df['usd'] >= self.min_usd) | (df['usd'].isna() & (df['tokens'].fillna(0) > 0))]
        for k, v in kept['etype'].value_counts().items():
            self.counts[k] = self.counts.get(k, 0) + int(v)

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=str(ROOT / 'config' / 'assets.json'))
    ap.add_argument('--wrappers', default=str(ROOT / 'config' / 'wrappers.json'))
    ap.add_argument('--morpho', default=str(ROOT / 'data' / 'morpho'))
    ap.add_argument('--lending', default=str(ROOT / 'data' / 'lending'))
    ap.add_argument('--prices', default=str(ROOT / 'data' / 'prices_llama_hourly.csv.gz'))
    ap.add_argument('--start', default='2023-01-01')
    ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--step-hours', type=int, default=24)
    ap.add_argument('--max-stale-hours', type=float, default=72)
    ap.add_argument('--min-edge-usd', type=float, default=10_000)
    ap.add_argument('--no-edges', action='store_true', help='skip edges.csv.gz (token features, nodes and flags only)')
    ap.add_argument('--out', default=str(ROOT / 'data' / 'graph'))
    args = ap.parse_args()

    _, assets = load_registry(args.registry)
    reg = {a['address'].lower(): a for a in assets}
    by_sym = {a['symbol']: a for a in assets}
    lower_sym = {a['symbol'].lower(): a['symbol'] for a in assets}
    tnode = {}                                    # token address -> node id (one shared string per token)

    def node_of(addr):
        if addr not in tnode:
            tnode[addr] = f"tok:{reg[addr]['symbol']}" if addr in reg else f'tok:{addr}'
        return tnode[addr]

    grid = np.arange(to_ts(args.start), to_ts(args.end) + 86400, args.step_hours * 3600, dtype='int64')
    stale = int(args.max_stale_hours * 3600)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = [f'# Exposure graph build\n\nSnapshots: {len(grid)} ({args.start} .. {args.end}, every {args.step_hours} h); '
              f'values older than {args.max_stale_hours:g} h count as missing.\n']
    ew = NoEdges(args.min_edge_usd) if args.no_edges else EdgeWriter(out / 'edges.csv.gz', args.min_edge_usd)

    prices = read_csv(args.prices, usecols=['symbol', 'ts', 'price'])
    if prices is not None:
        prices = prices.dropna().astype({'ts': 'int64', 'price': 'float64'})
        prices['symbol'] = prices['symbol'].astype(str).to_numpy(dtype=object)
        prices = prices.sort_values('ts', kind='stable').reset_index(drop=True)
    report.append(f"- prices: {'%d rows' % len(prices) if prices is not None else 'not found (token-unit values stay unconverted, no oracle gap)'}")

    token_meta, nodes = {}, {}
    mm, va, aa = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ---------- Morpho markets ----------
    mdir = Path(args.morpho)
    markets = read_csv(mdir / 'markets.csv', dtype={'market_id': str, 'collateral': str, 'loan': str})
    mnode = {}
    if markets is None or not len(markets):
        report.append('- Morpho: markets.csv not found, skipped')
    else:
        for c, s in (('collateral', 'collateral_symbol'), ('loan', 'loan_symbol')):
            markets[c] = markets[c].fillna('').str.lower()
            markets[s] = markets[s].fillna('').astype(str)
            for a, sym in zip(markets[c], markets[s]):
                token_meta.setdefault(a, sym)
        short = markets['market_id'].str[:14]
        full_ids = short.duplicated().any()
        mnode = dict(zip(markets['market_id'], 'mm:' + (markets['market_id'] if full_ids else short)))
        for r in markets.itertuples():
            nodes[mnode[r.market_id]] = {'ntype': 'morpho_market', 'symbol': f'{r.collateral_symbol}/{r.loan_symbol}',
                                         'address': r.market_id, 'lltv': r.lltv, 'oracle_class': r.oracle_class}
        parts = []
        cats = {'market_id': markets['market_id'], 'field': MARKET_FIELDS}
        for iv in ('day', 'hour'):
            h = read_obs(mdir / f'market_history_{iv}.csv.gz', cats, LAG[iv], iv == 'hour')
            if h is not None and len(h):
                parts.append(h)
                report.append(f'- Morpho market history ({iv}): {len(h)} points, {h.market_id.nunique()} markets, '
                              f'fields {sorted(h.field.unique())}')
        if parts:
            h = pd.concat(parts, ignore_index=True).sort_values('prio', kind='stable')
            h = h.drop_duplicates(['market_id', 'field', 'avail_ts'], keep='last')
            a = asof(h, ['market_id', 'field'], ['value'], grid, stale)
            del h, parts
            mm = wide_by_market(a, grid)
            del a
            for f in ('supplyAssetsUsd', 'borrowAssetsUsd', 'collateralAssetsUsd', 'collateralAssets'):
                if f not in mm:
                    mm[f] = np.nan
            info = markets.set_index('market_id')
            mm = mm[mm['market_id'].isin(info.index)].reset_index(drop=True)
            for c in ('collateral', 'loan', 'lltv', 'oracle_class'):
                mm[c] = mm['market_id'].map(info[c])
            unpriced = (mm['collateralAssetsUsd'] == 0) & (mm['collateralAssets'] > 0)   # Morpho had no price
            mm.loc[unpriced, 'collateralAssetsUsd'] = np.nan
            need = mm['collateralAssetsUsd'].isna() & mm['collateralAssets'].notna()
            if need.any() and prices is not None:   # token units -> USD with the market price at the snapshot
                syms = mm.loc[need, 'collateral'].map(lambda x: reg[x]['symbol'] if x in reg else '')
                mm.loc[need, 'collateralAssetsUsd'] = mm.loc[need, 'collateralAssets'].to_numpy() * \
                    price_lookup(prices, syms.to_numpy(), mm.loc[need, 't'].to_numpy())
            mm['col_node'] = mm['collateral'].map(node_of)
            mm['loan_node'] = mm['loan'].map(node_of)
            mm['mkt_node'] = mm['market_id'].map(mnode)

    # ---------- Morpho vaults ----------
    vaults = read_csv(mdir / 'vaults.csv', dtype={'vault': str, 'asset': str})
    if vaults is None or not len(vaults):
        report.append('- Morpho vaults: vaults.csv not found, skipped')
    else:
        vaults['asset'] = vaults['asset'].fillna('').str.lower()
        for a, sym in zip(vaults['asset'], vaults['asset_symbol'].fillna('').astype(str)):
            token_meta.setdefault(a, sym)
        short = vaults['vault'].str[:14]
        vnode = dict(zip(vaults['vault'], 'mv:' + (vaults['vault'] if short.duplicated().any() else short)))
        vasset = dict(zip(vaults['vault'], vaults['asset']))
        for r in vaults.itertuples():
            nodes[vnode[r.vault]] = {'ntype': 'morpho_vault', 'symbol': r.symbol if isinstance(r.symbol, str) else '',
                                     'address': r.vault, 'asset': node_of(r.asset)}
        parts = []
        mids = list(markets['market_id']) if markets is not None else []
        cats = {'vault': vaults['vault'], 'market_id': mids + [''], 'field': VAULT_FIELDS}
        for iv in ('day', 'hour'):
            v = read_obs(mdir / f'vault_allocation_{iv}.csv.gz', cats, LAG[iv], iv == 'hour', blank='market_id')
            if v is not None and len(v):
                parts.append(v)
                report.append(f'- Morpho vault history ({iv}): {len(v)} points, {v.vault.nunique()} vaults, '
                              f'fields {sorted(v.field.unique())}')
        if parts:
            v = pd.concat(parts, ignore_index=True)
            del parts
            unit = ~v['field'].str.endswith('Usd')
            if unit.any():       # token-unit fields -> USD at the observation time
                syms = v.loc[unit, 'vault'].map(vasset).map(lambda x: reg[x]['symbol'] if x in reg else '')
                v.loc[unit, 'value'] = v.loc[unit, 'value'].to_numpy() * price_lookup(prices, syms.to_numpy(), v.loc[unit, 'ts'].to_numpy())
                v['field'] = v['field'].str.replace(r'Assets$', 'AssetsUsd', regex=True)
            v = v.sort_values('prio', kind='stable').drop_duplicates(['vault', 'market_id', 'field', 'avail_ts'], keep='last')
            va = asof(v, ['vault', 'market_id', 'field'], ['value'], grid, stale)
            del v
            va = va[va['value'] > 0].reset_index(drop=True)
            for c in ('vault', 'market_id', 'field'):
                va[c] = decat(va[c])

    # ---------- Morpho data-quality rules, then the Morpho edges ----------
    if len(mm):
        mm, va, flags = clean_morpho(mm, va)
        if len(flags):
            info = markets.set_index('market_id')
            flags.insert(1, 'market', flags['market_id'].map(info['collateral_symbol']) + '/' + flags['market_id'].map(info['loan_symbol']))
            for c in ('first', 'last'):
                flags[c] = pd.to_datetime(flags[c], unit='s', utc=True).dt.strftime('%Y-%m-%dT%H:00Z')
        flags.to_csv(out / 'market_flags.csv', index=False)
        for rule, g in flags.groupby('rule') if len(flags) else []:
            report.append(f'- data-quality rule {rule}: {len(g)} markets ('
                          + ', '.join(f"{r.market} from {r.first[:10]}, borrow up to ${r.max_raw_borrow:,.0f} on paper, "
                                      f"${r.max_kept_borrow:,.0f} kept" for r in g.head(5).itertuples())
                          + (', ...' if len(g) > 5 else '') + '); see market_flags.csv')
        base = {'lltv': mm['lltv'], 'oracle_class': mm['oracle_class'], 'flag': mm['flag']}
        has_col = ~mm['collateral'].isin(['', ZERO])
        ew.write(pd.DataFrame({'t': mm['t'], 'src': mm['col_node'], 'dst': mm['mkt_node'], 'etype': 'collateral',
                               'usd': mm['collateralAssetsUsd'], 'tokens': mm['collateralAssets'], **base})[has_col])
        ew.write(pd.DataFrame({'t': mm['t'], 'src': mm['loan_node'], 'dst': mm['mkt_node'], 'etype': 'supply',
                               'usd': mm['supplyAssetsUsd'], **base}))
        ew.write(pd.DataFrame({'t': mm['t'], 'src': mm['mkt_node'], 'dst': mm['loan_node'], 'etype': 'borrow',
                               'usd': mm['borrowAssetsUsd'], **base}))
    if len(va):
        alloc = va['market_id'] != ''
        ew.write(pd.DataFrame({'t': va.loc[alloc, 't'], 'src': va.loc[alloc, 'vault'].map(vnode),
                               'dst': va.loc[alloc, 'market_id'].map(mnode), 'etype': 'allocation',
                               'usd': va.loc[alloc, 'value'], 'flag': va.loc[alloc, 'flag']}))
        ew.write(pd.DataFrame({'t': va.loc[~alloc, 't'], 'src': va.loc[~alloc, 'vault'].map(vasset).map(node_of),
                               'dst': va.loc[~alloc, 'vault'].map(vnode), 'etype': 'vault_asset', 'usd': va.loc[~alloc, 'value']}))

    # ---------- Aave v3 / Spark ----------
    ldir = Path(args.lending)
    ar = read_csv(ldir / 'aave_reserves.csv', dtype={'pool': str, 'asset': str, 'symbol': str, 'price_source': str})
    if ar is None or not len(ar):
        report.append('- Aave / Spark: aave_reserves.csv not found, skipped')
    else:
        ar['asset'] = ar['asset'].str.lower()
        for a, sym in ar[['asset', 'symbol']].drop_duplicates().itertuples(index=False):
            token_meta.setdefault(a, sym)
        if args.no_edges:      # features only: reserves outside every registry token's family are not needed
            wr_addr = {a.lower() for a in (json.loads(Path(args.wrappers).read_text()).get('by_address') or {})} \
                if Path(args.wrappers).exists() else set()
            def family_member(addr, sym):
                parts = re.split(r'[-_ ]', str(sym))
                return addr in reg or addr in wr_addr or (parts[0].upper() in DERIV_PREFIX and
                                                          any(x.lower() in lower_sym for x in parts[1:]))
            keep = {a for a, sym in ar[['asset', 'symbol']].drop_duplicates().itertuples(index=False) if family_member(a, sym)}
            ar = ar[ar['asset'].isin(keep)].reset_index(drop=True)
        em = read_csv(ldir / 'aave_emode.csv', dtype={'pool': str, 'collateral_bitmap': str, 'mode': str})
        ar['emode_ltv'] = emode_ltv(ar, em) if em is not None and len(em) else 0.0
        ar['avail_ts'] = ar['ts'].astype('int64')
        cols = ['supply', 'variable_debt', 'stable_debt', 'price_usd', 'ltv', 'liq_threshold', 'emode_ltv', 'frozen',
                'paused', 'supply_cap']
        aa = asof(ar, ['pool', 'asset'], cols, grid, stale)
        report.append(f'- Aave / Spark: {len(ar)} reserve reads, {ar.pool.nunique()} pools, {ar.asset.nunique()} assets'
                      + ('' if em is not None and len(em) else '; no e-mode file'))
        pools = ar['pool'].unique()
        del ar
        aa = aa.dropna(subset=['supply']).reset_index(drop=True)
        aa['supply_usd'] = aa['supply'] * aa['price_usd']
        aa['debt_usd'] = (aa['variable_debt'].fillna(0) + aa['stable_debt'].fillna(0)) * aa['price_usd']
        aa['node'] = aa['asset'].map(node_of)
        aa['frozen_any'] = ((aa['frozen'] > 0) | (aa['paused'] > 0)).astype(int)
        pnode = {p: f'pool:{p}' for p in pools}
        ew.write(pd.DataFrame({'t': aa['t'], 'src': aa['node'], 'dst': aa['pool'].map(pnode), 'etype': 'pool_supply',
                               'usd': aa['supply_usd'], 'tokens': aa['supply'], 'ltv': aa['ltv'],
                               'liq_threshold': aa['liq_threshold'], 'emode_ltv': aa['emode_ltv'], 'frozen': aa['frozen_any']}))
        ew.write(pd.DataFrame({'t': aa['t'], 'src': aa['pool'].map(pnode), 'dst': aa['node'], 'etype': 'pool_borrow',
                               'usd': aa['debt_usd'], 'frozen': aa['frozen_any']}))
        for p in pools:
            nodes[pnode[p]] = {'ntype': 'lending_pool', 'symbol': p}
    ew.close()

    # ---------- static edges and nodes ----------
    static = []
    wr = json.loads(Path(args.wrappers).read_text()) if Path(args.wrappers).exists() else {'pairs': []}
    for w, u in wr.get('pairs', []):
        if w in by_sym and u in by_sym:
            static.append((f'tok:{w}', f'tok:{u}', 'wrapper'))
    for a, u in (wr.get('by_address') or {}).items():
        if u in by_sym:
            static.append((node_of(a.lower()), f'tok:{u}', 'wrapper'))
    for a, sym in token_meta.items():
        if a in reg or not isinstance(sym, str):
            continue
        parts = re.split(r'[-_ ]', sym)
        if parts and parts[0].upper() in DERIV_PREFIX:
            hit = [lower_sym[p.lower()] for p in parts[1:] if p.lower() in lower_sym]
            if hit:
                static.append((node_of(a), f'tok:{hit[0]}', 'derivative'))
    static_df = pd.DataFrame(static, columns=['src', 'dst', 'etype']).drop_duplicates()
    static_df.to_csv(out / 'edges_static.csv', index=False)
    for a in assets:
        nodes[f"tok:{a['symbol']}"] = {'ntype': 'token', 'symbol': a['symbol'], 'address': a['address'].lower(),
                                       'category': a['category'], 'in_scope': int(a['in_scope'])}
    for a, sym in token_meta.items():
        if a and a != ZERO and a not in reg:
            nodes[node_of(a)] = {'ntype': 'token', 'symbol': sym, 'address': a, 'category': '', 'in_scope': 0}
    nd = pd.DataFrame([{'node_id': k, **v} for k, v in nodes.items()])
    nd = nd.reindex(columns=['node_id', 'ntype', 'symbol', 'address', 'category', 'in_scope', 'lltv', 'oracle_class', 'asset'])
    nd.to_csv(out / 'nodes.csv', index=False)
    report.append(f"\nNodes: {len(nd)} (" + ', '.join(f'{k} {v}' for k, v in nd['ntype'].value_counts().items()) + ')')
    report.append(f"Edges {'counted (--no-edges: not written)' if args.no_edges else 'written'}: {sum(ew.counts.values())} (of {ew.n_all} before the ${args.min_edge_usd:,.0f} floor); "
                  'by type: ' + ', '.join(f'{k} {v}' for k, v in ew.counts.items()))

    # ---------- token features ----------
    if len(mm):              # the edges are written: keep only what the features use
        mm = mm.drop(columns=[c for c in ('collateralAssets', 'loan', 'mkt_node', 'flag') if c in mm])
    feats = token_features(mm, va, aa, assets, static_df, grid, prices, args.min_edge_usd)
    feats.to_csv(out / 'token_features.csv.gz', index=False)
    report.append(f'Token features: {len(feats)} rows ({feats.symbol.nunique()} tokens x {len(grid)} snapshots)\n')
    report.append('| feature | tokens with a nonzero value | median of nonzero values |\n| --- | --- | --- |')
    for f in FEATURES:
        nz = feats.loc[feats[f].fillna(0) != 0]
        report.append(f'| {f} | {nz.symbol.nunique()} | {nz[f].median():,.3g} |' if len(nz) else f'| {f} | 0 | |')
    (out / 'build_report.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))


def token_features(mm, va, aa, assets, static_df, grid, prices, min_usd):
    syms = [a['symbol'] for a in assets if a['in_scope']]
    peg = {f"tok:{a['symbol']}": a['peg'] for a in assets}
    F = pd.DataFrame(index=pd.MultiIndex.from_product([syms, grid], names=['symbol', 't']), columns=list(FEATURES), dtype='float64')

    def put(series, name):
        """series indexed by (node, t) -> feature column (registry tokens only)."""
        if series is None or not len(series):
            return
        s = series.copy()
        s.index = pd.MultiIndex.from_arrays([s.index.get_level_values(0).str[4:], s.index.get_level_values(1)])
        s = s[s.index.get_level_values(0).isin(syms)]
        F.loc[s.index, name] = s.to_numpy(dtype='float64')

    direct = {}
    if len(mm):
        # collateral side: markets taking the token as collateral (USD-weighted features use known values only)
        m2 = mm.loc[~mm['collateral'].isin(['', ZERO]),
                    ['t', 'col_node', 'loan_node', 'collateralAssetsUsd', 'borrowAssetsUsd', 'lltv', 'oracle_class']]
        k2 = [m2['col_node'], m2['t']]
        c_usd = m2['collateralAssetsUsd']
        cu = c_usd.groupby(k2).sum(min_count=1)
        nz = cu.where(cu >= min_usd)                      # shares only where the token is meaningfully posted
        listed = (c_usd >= min_usd) | (c_usd.isna() & (m2['borrowAssetsUsd'] >= min_usd))
        put(listed.groupby(k2).sum(), 'mm_n_markets')
        put(cu, 'mm_collateral_usd')
        bu = m2['borrowAssetsUsd'].groupby(k2).sum(min_count=1)
        put(bu, 'mm_borrow_against_usd')
        put((m2['lltv'] * c_usd).groupby(k2).sum(min_count=1) / nz, 'mm_lltv_wavg')
        put((c_usd * m2['oracle_class'].isin(BLIND)).groupby(k2).sum(min_count=1) / nz, 'mm_blind_oracle_share')
        put((c_usd * (m2['oracle_class'] == 'custom')).groupby(k2).sum(min_count=1) / nz, 'mm_custom_oracle_share')
        cp, lp = m2['col_node'].map(peg), m2['loan_node'].map(peg)
        same = cp.notna() & (cp == lp)
        put(m2.loc[same, 'borrowAssetsUsd'].groupby([m2.loc[same, 'col_node'], m2.loc[same, 't']]).sum(min_count=1), 'mm_pegged_loop_usd')
        del m2, k2, c_usd, nz, listed, cp, lp, same
        # loan side
        put(mm['supplyAssetsUsd'].groupby([mm['loan_node'], mm['t']]).sum(min_count=1), 'mm_supply_usd')
        put(mm['borrowAssetsUsd'].groupby([mm['loan_node'], mm['t']]).sum(min_count=1), 'mm_borrowed_usd')
        direct['c'], direct['b'] = cu, bu
        if len(va):
            # allocations at (t, market) pairs the market history covers; a market's collateral never changes
            alloc = va.loc[va['market_id'] != '', ['t', 'vault', 'market_id', 'value']]
            mid = pd.Index(pd.unique(mm['market_id']))
            have = np.unique(mm['t'].to_numpy(dtype='int64') * len(mid) + mid.get_indexer(mm['market_id']))
            code = mid.get_indexer(alloc['market_id'])
            ok = (code >= 0) & np.isin(alloc['t'].to_numpy(dtype='int64') * len(mid) + code, have)
            alloc = alloc[ok]
            del have, code, ok
            col_of = mm.drop_duplicates('market_id').set_index('market_id')['col_node']
            alloc = alloc.assign(col_node=alloc['market_id'].map(col_of))
            ve = alloc['value'].groupby([alloc['col_node'], alloc['t']]).sum()
            put(ve, 'vault_exposure_usd')
            big = alloc[alloc['value'] >= min_usd]
            put(big['vault'].groupby([big['col_node'], big['t']]).nunique(), 'vault_n')
            direct['v'] = ve
            del alloc, big
    if len(aa):
        key = [aa['node'], aa['t']]
        put(aa['supply_usd'].groupby(key).sum(), 'aave_supply_usd')
        col = aa[aa['liq_threshold'] > 0]
        ac = col['supply_usd'].groupby([col['node'], col['t']]).sum()
        put(ac, 'aave_collateral_usd')
        ab = aa['debt_usd'].groupby(key).sum()
        put(ab, 'aave_borrow_usd')
        put(np.maximum(aa['ltv'], aa['emode_ltv']).groupby(key).max(), 'aave_ltv_max')
        put(aa['frozen_any'].groupby(key).max(), 'aave_frozen')
        capped = aa[aa['supply_cap'] > 0]
        put((capped['supply'] / capped['supply_cap']).groupby([capped['node'], capped['t']]).max(), 'aave_supply_cap_use')
        if prices is not None:
            listed = aa[aa['node'].str[4:].isin(syms) & (aa['supply_usd'] > 0)].copy()
            listed['mkt'] = price_lookup(prices, listed['node'].str[4:].to_numpy(), listed['t'].to_numpy())
            listed = listed.dropna(subset=['mkt', 'price_usd'])
            listed['gap_w'] = (listed['price_usd'] / listed['mkt'] - 1) * listed['supply_usd']
            k2 = [listed['node'], listed['t']]
            put(listed['gap_w'].groupby(k2).sum() / listed['supply_usd'].groupby(k2).sum(), 'aave_oracle_gap')
        direct['ac'], direct['ab'] = ac, ab

    # families: the token plus everything that wraps it or is derived from it (transitively)
    if direct:
        d = pd.concat({k: v for k, v in direct.items()}, axis=1).fillna(0.0)
        d.index = d.index.set_names(['node', 't'])
        d = d.reindex(columns=['c', 'b', 'v', 'ac', 'ab'], fill_value=0.0)
        children = {}
        for s, t in zip(static_df['src'], static_df['dst']):
            children.setdefault(t, set()).add(s)
        present = set(d.index.get_level_values(0))
        for s in syms:
            fam, todo = {f'tok:{s}'}, [f'tok:{s}']
            while todo:
                for c in children.get(todo.pop(), ()):
                    if c not in fam:
                        fam.add(c)
                        todo.append(c)
            fam &= present
            if not fam:
                continue
            sub = d[d.index.get_level_values(0).isin(fam)].groupby(level='t').sum()
            ix = pd.MultiIndex.from_arrays([[s] * len(sub), sub.index])
            F.loc[ix, 'family_collateral_usd'] = (sub['c'] + sub['ac']).to_numpy()
            F.loc[ix, 'family_vault_exposure_usd'] = sub['v'].to_numpy()
            F.loc[ix, 'family_borrow_against_usd'] = (sub['b'] + sub['ab']).to_numpy()

    F = F.reset_index()
    F.insert(2, 'time', pd.to_datetime(F['t'], unit='s', utc=True).dt.strftime('%Y-%m-%dT%H:00Z'))
    return F


if __name__ == '__main__':
    main()
