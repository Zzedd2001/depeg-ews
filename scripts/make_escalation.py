#!/usr/bin/env python3
"""Episode-level table for the escalation task: once a depeg has started, will it turn severe?

For every episode that passed make_labels' quality checks and every landmark L (default 1 and 6 hours
after the start), one row at the decision hour tau = start + L h, or at the hour the episode is known if a
gap in the data delays the sample that confirms it (common.known_ts), with two outcomes:
  y_severe  the episode's held depth (deepest level held for two consecutive samples, as in make_labels)
            reaches -5 %, i.e. severity major or collapse
  y_long    the episode lasts at least 24 h (start to its last hour outside half the threshold); only for
            landmarks below 23 h, where it is still open
An outcome is left empty when it is already decided at tau (held depth so far beyond -5 %) or unknown
because the episode was still open when the data ended.

Features at tau use only what is known at tau (the hour-tau price is observed by tau + 30 min):
  EP  depth so far as a fraction of the 5 % severity level, first-hour deviation, hours beyond the
      threshold so far, back inside the recovery band, run-up in the 24 h and 168 h before the start;
      the asset's earlier episodes (all closed before this one began), how many were severe or long,
      their deepest level, hours since the last one ended, asset age since its first price
  P, E the hourly feature groups of make_dataset.py at tau (deviation, market, wrappers, type; lending exposure)
Deviations in threshold units (1 % for USD pegs, 2 % for ETH), as in make_dataset.py.

Usage:
  python scripts/make_escalation.py                   # writes data/model/escalation.csv.gz (+ escalation_features.json)
  python scripts/make_escalation.py --landmarks 1 6 24
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
SEVERE = ('major', 'collapse')
LONG_HOURS = 24
PRIOR = (1.0, 4.0)          # Beta prior for an asset's share of severe / long episodes (mean 0.2), fixed in advance
EP_FEATURES = ['depth_frac', 'ep_z_first', 'ep_hours_below', 'ep_back_in_band', 'pre_z_min_24h', 'pre_z_mean_168h',
               'prior_episodes', 'prior_severe', 'prior_long', 'prior_severe_share', 'prior_long_share',
               'prior_max_depth_frac', 'hours_since_prior_end', 'asset_age_days', 'log_prior_episodes',
               'log_asset_age_days', 'und_below']


def held_depth(values):
    """Deepest level held for two consecutive known samples (make_labels' rule); NaN with fewer than two."""
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return np.nan
    return float(np.min(np.maximum(v[:-1], v[1:])))


def nan_stat(fn, v):
    v = v[~np.isnan(v)]
    return float(fn(v)) if len(v) else np.nan


def episode_rows(ep, dev, theta, landmarks, sev_level, band_frac, dev_obs=None):
    """ep: episodes of the target assets with int t0 / t1 and, optionally, tk (the hour each is known, from
    common.known_ts; default t0 + 1 h); dev: symbol -> deviation Series indexed by ts (the features read it);
    dev_obs: the same at hours with a price sample of their own (held depth is measured on it; default: dev)."""
    dev_obs = dev if dev_obs is None else dev_obs
    rows = []
    for sym, g in ep.groupby('symbol', sort=True):
        d = dev[sym]
        d_obs = dev_obs[sym]
        th = theta[sym]
        first_seen = d.dropna().index.min()
        n = n_sev = n_long = 0
        deepest, prev_end = 0.0, None
        for e in g.sort_values('t0').itertuples():
            severe = e.severity in SEVERE
            dur = (e.t1 - e.t0) / H + 1
            for L in landmarks:
                tau = max(e.t0 + L * H, int(getattr(e, 'tk', e.t0 + H)))     # never before the episode is known
                w = d.loc[e.t0:tau].to_numpy(dtype=float)
                kw = w[~np.isnan(w)]
                held = held_depth(d_obs.loc[e.t0:tau].to_numpy(dtype=float))
                decided = bool(held <= -sev_level)              # NaN compares False
                now = float(d.get(tau, np.nan))
                first = float(d.get(e.t0, np.nan))
                y_sev = np.nan if decided or (e.ongoing and not severe) else float(severe)
                y_long = np.nan if L >= LONG_HOURS - 1 or (e.ongoing and dur < LONG_HOURS) else float(dur >= LONG_HOURS)
                rows.append({
                    'episode_id': f'{sym}@{e.start}', 'symbol': sym, 'start': e.start, 't0': e.t0, 'landmark': L,
                    'ts': tau, 'severity': e.severity, 'held_min_dev': e.held_min_dev, 'duration_h': dur,
                    'ongoing': bool(e.ongoing), 'decided_severe': decided, 'y_severe': y_sev, 'y_long': y_long,
                    'depth_frac': max(0.0, -kw.min()) / sev_level if len(kw) else np.nan,
                    'ep_z_first': first / th,
                    'ep_hours_below': float((kw <= -th).sum()),
                    'ep_back_in_band': np.nan if np.isnan(now) else float(now > -th * band_frac),
                    'pre_z_min_24h': nan_stat(np.min, d.loc[e.t0 - 24 * H:e.t0 - H].to_numpy(dtype=float)) / th,
                    'pre_z_mean_168h': nan_stat(np.mean, d.loc[e.t0 - 168 * H:e.t0 - H].to_numpy(dtype=float)) / th,
                    'prior_episodes': float(n), 'prior_severe': float(n_sev), 'prior_long': float(n_long),
                    'prior_severe_share': (n_sev + PRIOR[0]) / (n + sum(PRIOR)),
                    'prior_long_share': (n_long + PRIOR[0]) / (n + sum(PRIOR)),
                    'prior_max_depth_frac': deepest,
                    'hours_since_prior_end': np.nan if prev_end is None else (tau - prev_end) / H,
                    'asset_age_days': (tau - first_seen) / 86400.0,
                })
            # this episode becomes history for the next one, which can only start after it closed
            # (24 h back inside the band), so its severity and length are known by then
            n += 1
            n_sev += severe
            n_long += dur >= LONG_HOURS
            held = float(e.held_min_dev)                      # NaN for a one-sample spike (it held no depth)
            if np.isfinite(held):
                deepest = max(deepest, max(0.0, -held) / sev_level)
            prev_end = e.t1
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    for name, default in (('registry', 'config/assets.json'), ('wrappers', 'config/wrappers.json'),
                          ('rules', 'config/label_rules.json'), ('labels', 'data/labels/labels_hourly.csv.gz'),
                          ('episodes', 'data/labels/episodes.csv'), ('prices', 'data/prices_llama_hourly.csv.gz'),
                          ('features', 'data/graph_hourly/token_features.csv.gz'),
                          ('out', 'data/model/escalation.csv.gz')):
        ap.add_argument(f'--{name}', default=str(ROOT / default))
    ap.add_argument('--landmarks', nargs='+', type=int, default=[1, 6])
    args = ap.parse_args()

    rules = json.loads(Path(args.rules).read_text())
    _, assets = load_registry(args.registry)
    meta = {a['symbol']: a for a in assets}
    sev_level = rules['severity_tiers'][0][0]                # below this depth an episode is 'minor'
    lab = pd.read_csv(args.labels, usecols=label_columns(args.labels))
    lab['ts'] = epoch_seconds(lab['hour'])
    lab['dev_obs'] = observed_only(lab)
    dev = {s: g.set_index('ts')['dev'].sort_index().clip(-1, 1) for s, g in lab.groupby('symbol')}
    dev_obs = {s: g.set_index('ts')['dev_obs'].sort_index().clip(-1, 1) for s, g in lab.groupby('symbol')}
    del lab
    theta = {s: rules['threshold'][meta[s]['peg']] for s in dev}
    ep = not_suspect(pd.read_csv(args.episodes))
    ep = ep[ep['symbol'].isin(list(dev))].copy()
    ep['t0'], ep['t1'] = epoch_seconds(ep['start']), epoch_seconds(ep['end'])
    ep['tk'] = np.asarray(known_ts(ep), dtype='int64')
    ep['ongoing'] = ep['ongoing'].astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
    rows = episode_rows(ep, dev, theta, sorted(args.landmarks), sev_level, rules['recovery_band_frac'], dev_obs)
    rows['row_id'] = np.arange(len(rows))

    feats, groups = md.build(args, rows=rows[['ts', 'symbol', 'row_id']])
    feats = feats.drop(columns=['ts', 'symbol', 'split'])
    table = rows.merge(feats, on='row_id', how='inner').drop(columns='row_id')
    table['log_prior_episodes'] = np.log1p(table['prior_episodes'])
    table['log_asset_age_days'] = np.log1p(table['asset_age_days'].clip(lower=0))
    table['und_below'] = (table['und_z_now'] <= -1).astype(float)
    tr, va = to_ts(rules['split']['train_end'][:16]), to_ts(rules['split']['valid_end'][:16])
    table['split'] = np.where(table['t0'] <= tr, 'train', np.where(table['t0'] <= va, 'valid', 'test'))
    table = table.sort_values(['landmark', 't0', 'symbol'], kind='stable').reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format='%.6g')
    (out.parent / 'escalation_features.json').write_text(json.dumps({'EP': EP_FEATURES, **groups}, indent=1))
    s = table.groupby(['landmark', 'split']).agg(
        episodes=('episode_id', 'size'), decided_severe=('decided_severe', 'sum'),
        severe_known=('y_severe', 'count'), severe=('y_severe', 'sum'),
        long_known=('y_long', 'count'), long=('y_long', 'sum'))
    print(f'wrote {out}: {len(table)} rows ({table["episode_id"].nunique()} episodes, {table["symbol"].nunique()} assets), '
          f'{len(EP_FEATURES)} EP, {len(groups["P"])} P, {len(groups["E"])} E features')
    print(s.astype(int).to_string())


if __name__ == '__main__':
    main()
