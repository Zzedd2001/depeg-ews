#!/usr/bin/env python3
"""Depeg episodes and early-warning labels from hourly prices (no manual annotation).

Deviation  d = value / reference - 1
  value      USD price (USD pegs) or price / WETH price (ETH pegs)
  reference  fixed 1.0 | 1.0 ETH (rebasing LSTs) | on-chain rate (forward-filled,
             never looks ahead) | trailing median of the previous 168 h
Episode    starts when d <= -threshold for >= 2 consecutive hourly samples, or a
           single sample at d <= -5%; closes after 24 consecutive samples back above
           -threshold/2, and ends at its last sample at or below -threshold/2. The sample that
           completes the start condition is 'confirmed_at' (the second sample of the run, or the
           -5% sample itself): the episode is known from then, or from start + 1 h if that is later
           (common.known_ts), which a gap in the data can delay by a few hours.
           Threshold 1% for USD pegs, 2% for ETH pegs. Episodes and their held
           depth use only hours with a price sample of their own: hours without one neither
           extend nor break a run. (Gaps of up to max_gap_fill_hours are carried forward in the
           deviation series that features read, marked obs = 0 in labels_hourly.csv.gz;
           --fill-counts restores v1.0, where carried-forward hours counted as samples.)
Labels     y_h(t) = 1 if an episode starts in (t, t+h], h in {24, 72};
           hours inside an episode or without data are left empty.

Usage:
  python scripts/make_labels.py --prices data/prices_llama_hourly.csv.gz
  python scripts/make_labels.py --prices data/dex_prices_hourly.csv.gz data/prices_llama_hourly.csv.gz \
      --primary dex --rates data/rates.csv.gz
  python scripts/make_labels.py ... --exclude-episodes data/labels/exclude_contradicted.csv --out-dir data/labels_robust
      (episodes listed there, by symbol and start, are flagged 'excluded': kept in episodes.csv, no labels)
Outputs (data/labels/): episodes.csv, labels_hourly.csv.gz, coverage.csv (with median_value_to_ref, the median of
value / reference: far from 1 means a wrong reference, e.g. a rate in the wrong unit), sensitivity.csv,
known_events_check.csv, checkpoint_report.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_registry  # noqa: E402

H = pd.Timedelta(hours=1)
RATE_BOUNDS = (0.5, 3.0)        # every registry rate (ETH or USD per token) lies well inside


def md(df, index=True):
    """Small markdown table writer (avoids the optional 'tabulate' dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for row in d.itertuples(index=False):
        lines.append('| ' + ' | '.join('' if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


# ---------------------------------------------------------------- loading
def collapse_dex_quotes(df):
    """DEX rows per asset, quote token and hour (step 03: n_trades, volume_quote, vwap, p25, p50, p75) ->
    one row per asset-hour. price = median across quote tokens of the within-hour median trade price,
    high = the same for the upper quartile; n_trades and volume_quote are summed. The median across
    quote tokens guards against one quote token depegging (USDC in March 2023); the within-hour
    quantiles guard against lending flows that pair the asset with a quote token at the loan-to-value
    ratio. Files with one row per asset-hour (a volume-weighted price) pass through with high = price."""
    if 'quote' not in df or 'p50' not in df:
        out = df.copy()
        if 'high' not in out:
            out['high'] = out['price']
        return out
    df = df.copy()
    for c in ('p50', 'p75', 'n_trades', 'volume_quote'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    out = (df.groupby(['symbol', 'hour'], sort=False)
             .agg(price=('p50', 'median'), high=('p75', 'median'), unit=('unit', 'first'),
                  n_trades=('n_trades', 'sum'), n_quotes=('quote', 'nunique'), volume_quote=('volume_quote', 'sum'))
             .reset_index())
    out['source'] = str(df['source'].iloc[0]) if 'source' in df and len(df) else 'dex'
    return out


def load_prices(paths, min_conf):
    frames = []
    for p in paths:
        df = collapse_dex_quotes(pd.read_csv(p))
        if 'source' not in df:
            df['source'] = Path(p).stem.split('.')[0]
        if 'confidence' in df:
            conf = pd.to_numeric(df['confidence'], errors='coerce')
            df = df[conf.isna() | (conf >= min_conf)]
        df['hour'] = pd.to_datetime(df['hour'], utc=True).dt.floor('h')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if 'unit' not in df:
            df['unit'] = 'USD'
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce') if 'ts' in df else np.nan
        frames.append(df[['symbol', 'hour', 'price', 'source', 'unit', 'ts']].dropna(subset=['symbol', 'hour', 'price']))
    return pd.concat(frames, ignore_index=True)


def load_rates(path):
    if not path:
        return None
    df = pd.read_csv(path)
    df = df[df['ok'] == 1].copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    df = df[df['rate'].between(RATE_BOUNDS[0], RATE_BOUNDS[1])]      # a reverted or mis-scaled read is not a rate
    return df[['symbol', 'time', 'rate']].dropna()


def load_exclusions(path):
    """(symbol, start) pairs of episodes to flag as 'excluded' (e.g. contradicted by a second price source)."""
    if not path:
        return set()
    df = pd.read_csv(path)
    return set(zip(df['symbol'], pd.to_datetime(df['start'], utc=True)))


# ---------------------------------------------------------------- deviation
def eth_ratio(prices, symbol, source, gap):
    """price(asset) / price(ETH). When raw timestamps exist (DefiLlama points can sit up to
    30 min from the hour), ETH is interpolated at the asset's own observation time, so a fast
    ETH move between the two observations does not show up as a fake discount."""
    a = prices[(prices['symbol'] == symbol) & (prices['source'] == source)]
    w = prices[(prices['symbol'] == 'WETH') & (prices['source'] == source)]
    if a.empty or w.empty:
        return pd.Series(dtype=float)
    if 'ts' in prices.columns and a['ts'].notna().all() and w['ts'].notna().all():
        w = w.sort_values('ts').drop_duplicates('ts')
        wt, wp = w['ts'].to_numpy(float), w['price'].to_numpy(float)
        at = a['ts'].to_numpy(float)
        i = np.searchsorted(wt, at)
        left = np.where(i > 0, at - wt[np.clip(i - 1, 0, len(wt) - 1)], np.inf)
        right = np.where(i < len(wt), wt[np.clip(i, 0, len(wt) - 1)] - at, np.inf)
        ok = (right == 0) | ((left <= 7200) & (right <= 7200))   # exact hit, or WETH on both sides within 2 h
        eth = np.where(ok, np.interp(at, wt, wp), np.nan)
        r = pd.Series(a['price'].to_numpy(float) / eth, index=pd.DatetimeIndex(a['hour']))
        r = r[~r.index.duplicated(keep='last')].sort_index().dropna()
    else:
        px, eth = series_for(prices, symbol, source), series_for(prices, 'WETH', source)
        if gap:
            px, eth = px.ffill(limit=gap), eth.ffill(limit=gap)
        r = (px / eth.reindex(px.index)).dropna()
    if r.empty:
        return r
    r = r.reindex(pd.date_range(r.index.min(), r.index.max(), freq='h'))
    return r.ffill(limit=gap) if gap else r


def series_for(prices, symbol, source):
    s = prices[(prices['symbol'] == symbol) & (prices['source'] == source)].set_index('hour')['price']
    s = s[~s.index.duplicated(keep='last')].sort_index()
    if s.empty:
        return s
    full = pd.date_range(s.index.min(), s.index.max(), freq='h')
    return s.reindex(full)


def observed_hours(asset, prices, source, index):
    """True at hours where the asset has a price sample of its own (for ETH pegs priced in USD: one with a WETH
    price within 2 h, see eth_ratio). Hours that deviation() fills by carrying the last value forward are False."""
    unit = (prices.loc[(prices['symbol'] == asset['symbol']) & (prices['source'] == source), 'unit']
            if 'unit' in prices else pd.Series(dtype=object))
    if asset['peg'] == 'ETH' and not (len(unit) and (unit == 'ETH').all()):
        own = eth_ratio(prices, asset['symbol'], source, 0)
    else:
        own = series_for(prices, asset['symbol'], source)
    return own.reindex(index).notna() if len(own) else pd.Series(False, index=index)


def deviation(asset, prices, rates, rules, source):
    """Hourly deviation series for one asset from one price source (NaN where unknown)."""
    gap = rules['max_gap_fill_hours']
    px = series_for(prices, asset['symbol'], source)
    px = px.ffill(limit=gap) if gap else px
    if px.empty:
        return px, 'no prices'
    unit = (prices.loc[(prices['symbol'] == asset['symbol']) & (prices['source'] == source), 'unit']
            if 'unit' in prices else pd.Series(dtype=object))
    if asset['peg'] == 'ETH' and len(unit) and (unit == 'ETH').all():
        value = px  # already quoted in ETH (DEX trades against WETH)
    elif asset['peg'] == 'ETH':
        value = eth_ratio(prices, asset['symbol'], source, gap)
        if value.empty:
            return value, 'no overlap with WETH prices'
    else:
        value = px
    kind = asset['reference']['kind']
    tm = value.shift(1).rolling(rules['trailing_median_hours'], min_periods=rules['trailing_median_min_hours']).median()
    if kind == 'fixed':
        ref = pd.Series(asset['reference'].get('value', 1.0), index=value.index)
    elif kind == 'eth_parity':
        ref = pd.Series(1.0, index=value.index)
    elif kind == 'rate':
        ref = tm.copy()
        kind = 'trailing_median (no rates)'
        if rates is not None:
            r = rates[rates['symbol'] == asset['symbol']].set_index('time')['rate'].sort_index()
            r = r[~r.index.duplicated(keep='last')]
            if not r.empty:
                # rate known at or before t only (as-of join), then fall back to the trailing median
                idx = r.index.searchsorted(value.index, side='right') - 1
                rv = np.where(idx >= 0, r.to_numpy()[np.clip(idx, 0, None)], np.nan)
                age = value.index - r.index[np.clip(idx, 0, None)]
                rv = np.where(age <= pd.Timedelta(hours=rules['rate_ffill_hours']), rv, np.nan)
                ref = pd.Series(rv, index=value.index).fillna(tm)
                kind = 'rate'
    else:  # trailing_median
        ref = tm
    return value / ref - 1.0, kind


# ---------------------------------------------------------------- episodes
def detect(dev, theta, min_run, severe, band_frac, recovery_hours):
    """Scan an hourly deviation series; returns a list of episode dicts."""
    eps, run_start, run_len, cur = [], None, 0, None
    rec = 0
    band = -theta * band_frac
    for t, d in dev.items():
        if np.isnan(d):
            continue  # unknown hours neither extend nor break a run
        if cur is None:
            if d <= -theta:
                if run_len == 0:
                    run_start = t
                run_len += 1
                if run_len >= min_run or d <= -severe:
                    cur = {'start': run_start, 'confirmed_at': t, 'min_dev': d, 'hours_below': run_len, 'last_bad': t}
                    rec = 0
            else:
                run_len, run_start = 0, None
        else:
            if d <= -theta:
                cur['hours_below'] += 1
            cur['min_dev'] = min(cur['min_dev'], d)
            if d > band:
                rec += 1
                if rec >= recovery_hours:
                    cur['end'] = cur['last_bad']
                    cur['ongoing'] = False
                    eps.append(cur)
                    cur, run_len, run_start, rec = None, 0, None, 0
            else:
                rec = 0
                cur['last_bad'] = t
    if cur is not None:
        cur['end'] = cur['last_bad']
        cur['ongoing'] = True
        eps.append(cur)
    for e in eps:
        e.pop('last_bad', None)
    return eps


def severity(min_dev, tiers):
    m = abs(min_dev)
    for cut, name in tiers:
        if m < cut:
            return name
    return tiers[-1][1]


def labels_for(dev, eps, horizons):
    out = pd.DataFrame({'dev': dev})
    inside = pd.Series(False, index=dev.index)
    for e in eps:
        inside.loc[e['start']:e['end']] = True
    out['in_episode'] = inside.astype(int)
    last = dev.dropna().index.max()
    for h in horizons:
        y = pd.Series(0.0, index=dev.index)
        for e in eps:
            y.loc[e['start'] - h * H:e['start'] - H] = 1.0
        y[inside | dev.isna()] = np.nan
        y[(y == 0) & (dev.index > last - h * H)] = np.nan  # right-censored
        out[f'y{h}'] = y
    return out


# ---------------------------------------------------------------- main
def run(args):
    rules = json.loads(Path(args.rules).read_text())
    q = rules.get('quality', {})
    snap_dev, snap_h = q.get('snapback_min_dev', -0.30), q.get('snapback_max_hours', 12)
    cluster_n = q.get('cluster_min_assets', 3)
    dead_dev, dead_h = q.get('dead_min_dev', -0.50), q.get('dead_min_hours_below', 24)
    _, assets = load_registry(args.registry)
    prices = load_prices(args.prices, rules['min_confidence'])
    rates = load_rates(args.rates)
    excluded = load_exclusions(getattr(args, 'exclude_episodes', ''))
    fill_counts = getattr(args, 'fill_counts', False)
    sources = list(dict.fromkeys(prices['source']))
    primary = args.primary or sources[0]
    secondary = [s for s in sources if s != primary]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = [a for a in assets if a['in_scope']]
    ycols = [f'y{h}' for h in rules['horizons_hours']]

    # pass 1: deviations and raw episodes per asset
    devs, cover, raw, keep_dev = {}, [], {}, {}
    for a in targets:
        # primary source first; an asset without primary data falls back to the next source
        used, dev, ref_kind = None, pd.Series(dtype=float), 'no prices'
        for src in [primary] + secondary:
            dev, ref_kind = deviation(a, prices, rates, rules, src)
            if not dev.empty and dev.notna().any():
                used = src
                break
        if used is None:
            cover.append({'symbol': a['symbol'], 'category': a['category'], 'status': ref_kind})
            continue
        others = [src for src in [primary] + secondary if src != used]
        # episodes use only hours with a sample of their own; carried-forward hours stay in `dev` for features
        obs = dev.notna() if fill_counts else (observed_hours(a, prices, used, dev.index) & dev.notna())
        dev_obs = dev.where(obs)
        known = dev_obs.dropna()
        steps = known.diff().abs().dropna()
        flat = float((steps < 1e-7).mean()) if len(steps) else 1.0
        derived = flat >= rules['flat_share_flag']
        devs[a['symbol']] = dev_obs
        cover.append({'symbol': a['symbol'], 'category': a['category'], 'status': 'ok', 'source': used, 'reference': ref_kind,
                      'first_hour': known.index.min(), 'last_hour': known.index.max(), 'n_hours': len(known),
                      'coverage': round(len(known) / len(dev), 3), 'coverage_filled': round(float(dev.notna().mean()), 3),
                      'flat_share': round(flat, 3), 'derived_suspect': derived,
                      'median_value_to_ref': round(float(np.median(known.to_numpy() + 1.0)), 4)})
        theta = rules['threshold'][a['peg']]
        eps = detect(dev_obs, theta, rules['min_consecutive_hours'], rules['single_hour_severe'],
                     rules['recovery_band_frac'], rules['recovery_hours'])
        second = [deviation(a, prices, rates, rules, src)[0] for src in others]
        lst = []
        for e in eps:
            conf = ''
            for d2 in second:
                if not d2.empty:
                    w = d2.loc[e['start'] - H:e['end'] + H]
                    conf = 'yes' if (w <= -theta * rules['recovery_band_frac']).any() else ('no' if w.notna().any() else conf)
            win = dev_obs.loc[e['start']:e['end']].dropna().to_numpy(dtype=float)
            # deepest level held for two consecutive known samples: one bad print inside a long, mild
            # episode must not make it a 'collapse', nor may the last sample count on its own; a
            # one-sample spike (it opens an episode when beyond -5 %) holds no depth, so it is minor
            # (its depth stays in min_dev)
            held = float(np.min(np.maximum(win[:-1], win[1:]))) if len(win) > 1 else np.nan
            sev = severity(held, rules['severity_tiers']) if np.isfinite(held) else rules['severity_tiers'][0][1]
            lst.append({'symbol': a['symbol'], 'category': a['category'], 'peg': a['peg'], 'start': e['start'],
                        'end': e['end'], 'confirmed_at': e['confirmed_at'], 'hours_below': e['hours_below'],
                        'min_dev': round(e['min_dev'], 5),
                        'held_min_dev': round(float(held), 5), 'hours_beyond_dead': int((win <= dead_dev).sum()),
                        'severity': sev, 'ongoing': e['ongoing'],
                        'source': used, 'confirmed_by_second_source': conf, 'derived_suspect': derived,
                        'suspect': '', 'terminal': False})
        raw[a['symbol']] = lst
        keep_dev[a['symbol']] = (dev, obs)

    # pass 2: data-quality flags. A deep drop that is fully gone within hours is far more often a bad
    # price print than a depeg; one-hour drops hitting several assets in the same hour likewise.
    # Flagged episodes stay in episodes.csv but give no labels (their hours and run-up are left empty).
    singles = {}
    for lst in raw.values():
        for e in lst:
            if e['hours_below'] == 1:
                singles[e['start']] = singles.get(e['start'], 0) + 1
    for lst in raw.values():
        for e in lst:
            dur = (e['end'] - e['start']).total_seconds() / 3600 + 1
            if e['min_dev'] <= snap_dev and not e['ongoing'] and dur <= snap_h:
                e['suspect'] = 'snapback'
            elif e['hours_below'] == 1 and singles.get(e['start'], 0) >= cluster_n:
                e['suspect'] = 'same_hour_cluster'
            elif (e['symbol'], pd.Timestamp(e['start'])) in excluded:
                e['suspect'] = 'excluded'

    # pass 3: collapsed assets stop being prediction targets; labels
    episodes, label_frames, dead, dropped = [], [], {}, 0
    for sym, lst in raw.items():
        dev, obs = keep_dev[sym]
        death = next((e['start'] for e in lst if not e['suspect'] and e['hours_beyond_dead'] >= dead_h), None)
        kept = []
        for e in lst:
            if death is not None and e['start'] > death:
                dropped += 1          # swings of a token that already collapsed are not new depegs
                continue
            e['terminal'] = death is not None and e['start'] == death
            kept.append(e)
        if death is not None:
            dead[sym] = death
        episodes += kept
        lab = labels_for(dev, [e for e in kept if not e['suspect']], rules['horizons_hours'])
        for e in kept:
            if e['suspect']:
                for h, c in zip(rules['horizons_hours'], ycols):
                    lab.loc[e['start'] - h * H:e['end'], c] = np.nan
        if death is not None:
            lab.loc[death:, ycols] = np.nan
        lab['obs'] = obs.reindex(lab.index).fillna(False).astype(int)
        lab.insert(0, 'symbol', sym)
        label_frames.append(lab)

    ep = pd.DataFrame(episodes, columns=['symbol', 'category', 'peg', 'start', 'end', 'confirmed_at', 'hours_below', 'min_dev',
                                         'held_min_dev', 'hours_beyond_dead', 'severity', 'ongoing', 'source',
                                         'confirmed_by_second_source', 'derived_suspect', 'suspect', 'terminal'])
    cov = pd.DataFrame(cover)
    ep.to_csv(out / 'episodes.csv', index=False)
    cov.to_csv(out / 'coverage.csv', index=False)
    if label_frames:
        lab = pd.concat(label_frames)
        lab.index.name = 'hour'
        lab.reset_index().to_csv(out / 'labels_hourly.csv.gz', index=False, float_format='%.6g')

    # sensitivity grid: raw detection counts before the quality filters (derived series excluded)
    sens = []
    use = {s: d for s, d in devs.items() if not cov.set_index('symbol').loc[s, 'derived_suspect']}
    peg = {a['symbol']: a['peg'] for a in targets}
    for tu in rules['sensitivity']['USD']:
        for te in rules['sensitivity']['ETH']:
            for mr in rules['sensitivity']['min_consecutive_hours']:
                n = sum(len(detect(d, tu if peg[s] == 'USD' else te, mr, rules['single_hour_severe'],
                                   rules['recovery_band_frac'], rules['recovery_hours'])) for s, d in use.items())
                sens.append({'theta_usd': tu, 'theta_eth': te, 'min_consecutive_hours': mr, 'episodes': n})
    sens = pd.DataFrame(sens)
    sens.to_csv(out / 'sensitivity.csv', index=False)

    # known events, checked against episodes that passed the quality filters
    good = ep[ep['suspect'] == ''] if len(ep) else ep
    kn = []
    for k in json.loads(Path(args.known).read_text()):
        day = pd.Timestamp(k['date'], tz='UTC')
        hit = good[(good['symbol'] == k['symbol']) & (good['start'] <= day + pd.Timedelta(days=3)) &
                   (good['end'] >= day - pd.Timedelta(days=3))] if len(good) else good
        has = k['symbol'] in devs
        kn.append({'symbol': k['symbol'], 'date': k['date'], 'expect': k['expect'],
                   'data': 'yes' if has and devs[k['symbol']].loc[day - 3 * 24 * H: day + 3 * 24 * H].notna().any() else 'no',
                   'detected': 'yes' if len(hit) else 'no',
                   'min_dev': round(float(hit['min_dev'].min()), 4) if len(hit) else '',
                   'start': hit['start'].min() if len(hit) else '', 'note': k['note']})
    kn = pd.DataFrame(kn)
    kn.to_csv(out / 'known_events_check.csv', index=False)
    report(out, rules, ep, cov, sens, kn, primary, secondary, dead, dropped)
    print((out / 'checkpoint_report.md').read_text())


def report(out, rules, ep, cov, sens, kn, primary, secondary, dead=None, dropped=0):
    L = ['# Checkpoint report: depeg episodes', '',
         f'Primary price source: `{primary}`; second source: {", ".join(secondary) or "none"}.',
         f"Rules: USD threshold {rules['threshold']['USD']:.1%}, ETH threshold {rules['threshold']['ETH']:.1%}, "
         f"at least {rules['min_consecutive_hours']} consecutive hours or one hour beyond {rules['single_hour_severe']:.0%}; "
         f"recovery after {rules['recovery_hours']} h back within half the threshold.", '']
    ok = cov[cov['status'] == 'ok'] if len(cov) else cov
    flat = ', '.join(ok[ok['derived_suspect'].astype(bool)]['symbol']) if len(ok) else ''
    L += [f"Assets with data: {len(ok)} of {len(cov)}; suspected derived (flat) series excluded from counts: "
          f"{flat or 'none'}.", '']
    dead = dead or {}
    sus = ep[ep['suspect'] != ''] if len(ep) else ep
    use = ep[(~ep['derived_suspect'].astype(bool)) & (ep['suspect'] == '')] if len(ep) else ep
    total = len(use)
    L += [f'## Episodes: {total}', '',
          f'Left out of the counts and the labels: {len(sus)} flagged episodes '
          f"({', '.join(f'{k} {v}' for k, v in sus['suspect'].value_counts().items()) if len(sus) else 'none'}), "
          f"{int((ep['derived_suspect'].astype(bool) & (ep['suspect'] == '')).sum()) if len(ep) else 0} episodes on series "
          f'flagged as derived, and {dropped} swings of assets after they had already collapsed. Details at the end.', '']
    if total:
        use = use.assign(year=pd.to_datetime(use['start']).dt.year)
        L += ['By year and category:', '', md(pd.crosstab(use['year'], use['category'], margins=True)), '']
        L += ['By severity:', '', md(use['severity'].value_counts().to_frame('episodes')), '']
        tr = pd.Timestamp(rules['split']['train_end'])
        va = pd.Timestamp(rules['split']['valid_end'])
        st = pd.to_datetime(use['start'])
        L += [f"Split: train {int((st <= tr).sum())}, validation {int(((st > tr) & (st <= va)).sum())}, test {int((st > va).sum())}.", '']
        per = use.groupby('symbol').size().sort_values(ascending=False)
        L += [f"Concentration: the 5 assets with the most episodes ({', '.join(per.head(5).index)}) account for "
              f"{per.head(5).sum() / total:.0%} of all episodes. Evaluate against a per-asset base-rate baseline and "
              f"report asset-balanced metrics, or a model can score well by learning which assets deviate often.", '']
        L += ['Episodes per asset:', '', md(per.to_frame('episodes')), '']
    need = rules['checkpoint_min_episodes']
    if total >= need:
        verdict = f'PASS: {total} episodes >= {need}. Keep the default rules.'
    else:
        best = sens.sort_values('episodes', ascending=False).head(3).to_dict('records')
        verdict = (f'BELOW TARGET: {total} episodes < {need}. Options: lower the USD threshold to 0.5%, '
                   f'add Arbitrum/Base assets, or keep the count and report per-event results. '
                   f'Most permissive grid settings: {best}')
    exp = kn[kn['expect'].astype(str).str.startswith('onchain')] if len(kn) else kn
    known_line = (f"Expected on-chain events detected: {int((exp['detected'] == 'yes').sum())} of {len(exp)}; "
                  f"{int((exp['data'] == 'no').sum())} had no price data around the date.") if len(exp) else ''
    qrows = sus[['symbol', 'start', 'end', 'hours_below', 'min_dev', 'suspect']] if len(sus) else sus
    deadrows = pd.DataFrame([{'symbol': k, 'collapsed_from': v} for k, v in sorted(dead.items(), key=lambda x: x[1])])
    L += ['## Checkpoint', '', verdict, '', '## Known events', '', known_line, '', md(kn, index=False), '',
          '## Data-quality filters', '',
          'Suspected bad price prints (snapback = a drop of 30% or more that is gone within 12 hours; '
          'same_hour_cluster = one-hour drops in three or more assets in the same hour). Check these against DEX prices:', '',
          md(qrows, index=False) if len(qrows) else 'none', '',
          'Assets that collapsed (at least 24 hours beyond -50%); they get no labels afterwards:', '',
          md(deadrows, index=False) if len(deadrows) else 'none', '',
          '## Sensitivity (raw episode counts, before the quality filters)', '', md(sens.pivot_table(index=['theta_usd', 'theta_eth'],
          columns='min_consecutive_hours', values='episodes')), '']
    (out / 'checkpoint_report.md').write_text('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prices', nargs='+', default=[str(ROOT / 'data' / 'prices_llama_hourly.csv.gz')])
    ap.add_argument('--primary', default='', help="source name used for labels, e.g. 'dex'; default: first file's source")
    ap.add_argument('--rates', default='')
    ap.add_argument('--registry', default=str(ROOT / 'config' / 'assets.json'))
    ap.add_argument('--rules', default=str(ROOT / 'config' / 'label_rules.json'))
    ap.add_argument('--known', default=str(ROOT / 'config' / 'known_events.json'))
    ap.add_argument('--out-dir', default=str(ROOT / 'data' / 'labels'))
    ap.add_argument('--exclude-episodes', default='', help='CSV with symbol,start of episodes to flag as excluded')
    ap.add_argument('--fill-counts', action='store_true',
                    help='v1.0 behaviour: hours carried forward over gaps count as samples for episodes and held depth')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
