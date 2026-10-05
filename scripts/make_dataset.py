#!/usr/bin/env python3
"""Hourly modeling table: causal features for every asset-hour that has a label.

Inputs:
  data/labels/labels_hourly.csv.gz          hour, symbol, dev, in_episode, y24, y72 (make_labels.py)
  data/labels/episodes.csv                  episode starts, for history and contagion counts
  data/prices_llama_hourly.csv.gz           WETH price, for market context
  data/graph_hourly/token_features.csv.gz   hourly exposure features (build_graph.py --step-hours 1 --no-edges)
  config/assets.json, config/wrappers.json

Every feature at hour t uses only information available by t:
  - deviations up to hour t (the hour-t price is observed at most 30 min after t, while a label
    concerns episodes starting at t+1 h or later, judged on prices observed after t+30 min);
  - an episode counts from the hour it is known (common.known_ts): one hour after its start, when its
    second sample confirms it, or later when a gap delays that sample;
  - exposure features are already as-of with a 1 h lag (build_graph.py).
Deviation features are in units of the asset's label threshold (1 % for USD pegs, 2 % for ETH),
so USD and ETH assets share one scale.

Feature groups (written to dataset_features.json):
  P  deviation, episode history, market context, underlying of wrappers, asset type
  E  lending exposure (Morpho, Aave / Spark) and its changes

Usage:
  python scripts/make_dataset.py           # writes data/model/dataset.pkl.gz (+ dataset_features.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, epoch_seconds, known_ts, load_registry, not_suspect, to_ts  # noqa: E402

H = 3600
EXPOSURE_USD = ['mm_collateral_usd', 'mm_borrow_against_usd', 'mm_pegged_loop_usd', 'mm_supply_usd', 'mm_borrowed_usd',
                'vault_exposure_usd', 'family_collateral_usd', 'family_vault_exposure_usd', 'family_borrow_against_usd',
                'aave_supply_usd', 'aave_collateral_usd', 'aave_borrow_usd']
EXPOSURE_RAW = ['mm_n_markets', 'mm_lltv_wavg', 'mm_blind_oracle_share', 'mm_custom_oracle_share', 'vault_n',
                'aave_ltv_max', 'aave_frozen', 'aave_supply_cap_use', 'aave_oracle_gap']
EXPOSURE_CHANGE = ['mm_borrow_against_usd', 'mm_supply_usd', 'vault_exposure_usd', 'family_collateral_usd',
                   'family_borrow_against_usd', 'aave_supply_usd', 'aave_borrow_usd']
CATEGORIES = ['synthetic', 'rwa_backed', 'yield_bearing', 'cdp', 'lst', 'lrt']


def hours_since(mask):
    """Hours since the last True in each column of a boolean frame (NaN before the first)."""
    idx = np.arange(len(mask), dtype=float)[:, None]
    last = pd.DataFrame(np.where(mask.to_numpy(), idx, np.nan), index=mask.index, columns=mask.columns).ffill()
    return idx - last


def deviation_features(z):
    """z: hours x assets frame of threshold-normalised deviations (complete hourly index)."""
    f = {'z_now': z}
    for w in (6, 24, 72, 168):
        f[f'z_min_{w}h'] = z.rolling(w, min_periods=1).min()
    f['z_max_24h'] = z.rolling(24, min_periods=1).max()
    for w in (24, 168):
        f[f'z_mean_{w}h'] = z.rolling(w, min_periods=1).mean()
        f[f'z_std_{w}h'] = z.rolling(w, min_periods=6).std()
    for k in (1, 6, 24):
        f[f'z_chg_{k}h'] = z - z.shift(k)
    f['z_vs_mean_168h'] = z - f['z_mean_168h']
    near = (z < -0.5).astype(float).where(z.notna())
    f['near_miss_168h'] = near.rolling(168, min_periods=1).sum()
    f['near_frac_24h'] = near.rolling(24, min_periods=1).mean()
    f['hours_since_near_miss'] = hours_since(near.fillna(0) > 0).clip(upper=24 * 90)
    f['missing_frac_24h'] = z.isna().astype(float).rolling(24, min_periods=1).mean()
    return f


def episode_features(starts, index, symbols):
    """starts: DataFrame symbol, ts with the hour each episode is known (common.known_ts), not its start."""
    conf = pd.DataFrame(0.0, index=index, columns=symbols)
    for sym, ts in zip(starts['symbol'], starts['ts']):
        if sym in conf.columns and ts in conf.index:
            conf.loc[ts, sym] += 1
    f = {}
    for days in (30, 90, 365):
        f[f'episodes_{days}d'] = conf.rolling(24 * days, min_periods=1).sum()
    f['hours_since_episode'] = hours_since(conf > 0).clip(upper=24 * 365)
    allc = conf.sum(axis=1)
    market = {'mkt_episodes_24h': allc.rolling(24, min_periods=1).sum(), 'mkt_episodes_168h': allc.rolling(168, min_periods=1).sum()}
    return f, market


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=str(ROOT / 'config' / 'assets.json'))
    ap.add_argument('--wrappers', default=str(ROOT / 'config' / 'wrappers.json'))
    ap.add_argument('--rules', default=str(ROOT / 'config' / 'label_rules.json'))
    ap.add_argument('--labels', default=str(ROOT / 'data' / 'labels' / 'labels_hourly.csv.gz'))
    ap.add_argument('--episodes', default=str(ROOT / 'data' / 'labels' / 'episodes.csv'))
    ap.add_argument('--prices', default=str(ROOT / 'data' / 'prices_llama_hourly.csv.gz'))
    ap.add_argument('--features', default=str(ROOT / 'data' / 'graph_hourly' / 'token_features.csv.gz'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'model' / 'dataset.pkl.gz'))
    return ap.parse_args(argv)


def build(args, rows=None):
    """Feature table for asset-hours; returns (table, {'P': [...], 'E': [...]}).
    rows: DataFrame with int 'ts' (unix seconds, whole hours) and 'symbol'; its other columns pass through.
    None = every asset-hour with an onset label (the hourly modeling table). Rows outside the hours or
    assets of labels_hourly are dropped."""
    rules = json.loads(Path(args.rules).read_text())
    _, assets = load_registry(args.registry)
    meta = {a['symbol']: a for a in assets}
    lab = pd.read_csv(args.labels)
    lab['ts'] = epoch_seconds(lab['hour'])
    symbols = sorted(lab['symbol'].unique())
    index = pd.Index(np.arange(lab['ts'].min(), lab['ts'].max() + H, H, dtype='int64'), name='ts')
    thr = pd.Series({s: rules['threshold'][meta[s]['peg']] for s in symbols})
    dev = lab.pivot(index='ts', columns='symbol', values='dev').reindex(index=index, columns=symbols)
    z = dev.clip(-1, 1) / thr
    del dev

    # P: deviations, history, market context, wrappers, type
    feats = deviation_features(z)
    ep = not_suspect(pd.read_csv(args.episodes))
    ep['ts'] = known_ts(ep)
    hist, market = episode_features(ep[['symbol', 'ts']], index, symbols)
    feats.update(hist)
    usd = [s for s in symbols if meta[s]['peg'] == 'USD']
    market['mkt_stress_frac'] = (z < -0.5).sum(axis=1) / z.notna().sum(axis=1).replace(0, np.nan)
    market['mkt_mean_z_usd'] = z[usd].clip(-5, 5).mean(axis=1)
    px = pd.read_csv(args.prices, usecols=['symbol', 'ts', 'price'])
    weth = px[px['symbol'] == 'WETH'].sort_values('ts')
    del px
    eth = pd.merge_asof(pd.DataFrame({'ts': index.to_numpy()}), weth[['ts', 'price']].astype({'ts': 'int64'}),
                        on='ts', direction='backward', tolerance=2 * H).set_index('ts')['price']
    leth = np.log(eth)
    for k in (1, 24, 168):
        market[f'eth_ret_{k}h'] = leth - leth.shift(k)
    market['eth_vol_24h'] = (leth - leth.shift(1)).rolling(24, min_periods=12).std()
    for name, s in market.items():
        feats[name] = pd.DataFrame(np.repeat(s.to_numpy()[:, None], len(symbols), axis=1), index=index, columns=symbols)
    wr = json.loads(Path(args.wrappers).read_text()) if Path(args.wrappers).exists() else {'pairs': []}
    under = {w: u for w, u in wr.get('pairs', []) if w in symbols and u in symbols}
    for name, src in (('und_z_now', feats['z_now']), ('und_z_min_24h', feats['z_min_24h'])):
        feats[name] = pd.DataFrame({s: (src[under[s]] if s in under else np.nan) for s in symbols}, index=index)
    if rows is None:
        picked = lab.loc[lab['y24'].notna() | lab['y72'].notna(), ['ts', 'symbol', 'y24', 'y72', 'in_episode']]
    else:
        ok = rows['symbol'].isin(symbols) & rows['ts'].isin(index)
        if (~ok).any():
            print(f'build: {int((~ok).sum())} of {len(rows)} requested asset-hours are outside the data and are dropped')
        picked = rows[ok]
    picked = picked.sort_values(['ts', 'symbol'], kind='stable')
    passthrough = list(picked.columns)
    del lab, z
    ri, ci = index.get_indexer(picked['ts']), pd.Index(symbols).get_indexer(picked['symbol'])
    long = pd.DataFrame({c: picked[c].to_numpy() for c in picked.columns})
    del picked
    for name in list(feats):               # pick each asset-hour out of the hours x assets frames, freeing as we go
        long[name] = feats.pop(name).to_numpy(dtype='float64')[ri, ci].astype('float32')
    long['peg_eth'] = long['symbol'].map(lambda s: float(meta[s]['peg'] == 'ETH'))
    for c in CATEGORIES:
        long[f'cat_{c}'] = long['symbol'].map(lambda s, c=c: float(meta[s]['category'] == c))

    # E: exposure
    cols = EXPOSURE_USD + EXPOSURE_RAW
    tf = pd.read_csv(args.features, usecols=['symbol', 't'] + cols, dtype={c: 'float32' for c in cols})
    tf = tf.rename(columns={'t': 'ts'}).sort_values(['symbol', 'ts']).reset_index(drop=True)
    ex = tf[['symbol', 'ts']].copy()
    for c in EXPOSURE_USD:
        ex[f'log_{c}'] = np.log1p(tf[c].clip(lower=0))
    for c in EXPOSURE_RAW:
        ex[c] = tf[c]
    for c in EXPOSURE_CHANGE:
        lc = ex[f'log_{c}']
        for k in (24, 72, 168):
            ex[f'dlog_{c}_{k}h'] = lc - lc.groupby(tf['symbol'], sort=False).shift(k)
    mc = tf['mm_collateral_usd'].where(tf['mm_collateral_usd'] >= 1e4)          # ratios only where at least $10k
    fc = tf['family_collateral_usd'].where(tf['family_collateral_usd'] >= 1e4)  # is posted: tiny denominators are noise
    ex['borrow_to_collateral'] = tf['mm_borrow_against_usd'] / mc
    ex['family_borrow_to_collateral'] = tf['family_borrow_against_usd'] / fc
    del tf
    ex = ex.set_index(['symbol', 'ts'])
    ex_cols = list(ex.columns)
    aligned = ex.reindex(pd.MultiIndex.from_arrays([long['symbol'], long['ts']]))   # the asset-hours of `long`, in order
    del ex
    for c in ex_cols:
        long[c] = aligned[c].to_numpy(dtype='float32')
    del aligned
    has = (long['log_family_collateral_usd'].fillna(0) > 0) | (long['log_mm_supply_usd'].fillna(0) > 0) | \
          (long['log_aave_supply_usd'].fillna(0) > 0)
    long['has_exposure'] = has.astype(float)

    # splits
    tr, va = to_ts(rules['split']['train_end'][:16]), to_ts(rules['split']['valid_end'][:16])
    long['split'] = np.where(long['ts'] <= tr, 'train', np.where(long['ts'] <= va, 'valid', 'test'))
    exposure = ex_cols + ['has_exposure']
    keep = set(passthrough) | {'split'}
    price = [c for c in long.columns if c not in keep and c not in exposure]
    for c in price + exposure:
        if long[c].dtype != np.float32:
            long[c] = long[c].astype('float32')
    return long, {'P': price, 'E': exposure}


def main():
    args = parse_args()
    long, groups = build(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_pickle(out, compression={'method': 'gzip', 'compresslevel': 1})
    (out.parent / 'dataset_features.json').write_text(json.dumps(groups, indent=1))
    n = long.groupby('split').agg(rows=('symbol', 'size'), y24=('y24', 'sum'), y72=('y72', 'sum'), assets=('symbol', 'nunique'))
    print(f'wrote {out}: {len(long):,} asset-hours, {len(groups["P"])} P features, {len(groups["E"])} E features')
    print(n.to_string())


if __name__ == '__main__':
    main()
