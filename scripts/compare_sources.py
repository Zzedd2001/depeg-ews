#!/usr/bin/env python3
"""Check rule-based depeg episodes against DEX trade prices (a second, separately built price source).

For every episode in <labels>/episodes.csv (labels from DefiLlama prices), take the asset's DEX trades
from 2 h before the start until 72 h into the episode (or its end, if sooner), and measure each traded
hour against the same reference value the labels use (fixed peg, ETH, or the on-chain exchange rate
when --rates is given). DEX prices come from transfer pairs (sql/), so lending flows can look like
trades at the loan-to-value ratio or the liquidation discount. Hence two hourly values: the median
trade price ("mid") and the upper quartile ("high": three quarters of the hour's trades at or below
it); for USD-pegged assets each is the median across quote tokens.
  confirmed     some hour with at least --min-hour-trades trades has high at or beyond -threshold
  partial       the same, but high only reaches -threshold/2
  contradicted  at least 10 trades over at least 3 hours (--contra-trades, --contra-hours), and mid within
                threshold/2 of the reference, above or below, in every traded hour
  unverified    anything else (too few trades, no DEX prices, or dips only in thin hours)
Hours without trades are never filled in.

Placebo windows: the same check on the same asset's DEX prices at the episode's window shifted by
+-7, 14, 21 and 28 days, where no labelled episode lies within a day. The share confirmed there is
what the check reports when nothing happened.

The script also lists DEX-only episodes (episodes of the robust 'high' series, with at least two hours
beyond the threshold in their first 24 hours, and no labelled episode of the asset within a day) and
checks the known events.

Writes <labels>/source_check.csv (one row per episode), source_check_placebo.csv,
source_check_dex_only.csv, source_check.md, and exclude_contradicted.csv: contradicted episodes not
already flagged, for
  python scripts/make_labels.py ... --exclude-episodes <labels>/exclude_contradicted.csv

Usage:
  python scripts/compare_sources.py --dex data/dex_prices_hourly.csv --rates data/rates.csv.gz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_labels as ml  # noqa: E402
from common import ROOT, load_registry  # noqa: E402

H = pd.Timedelta(hours=1)
D = pd.Timedelta(days=1)
BEFORE_H, CHECK_H, AFTER_H = 2, 72, 2
MIN_TRADES, MIN_HOURS = 10, 3          # evidence a contradiction needs (defaults of --contra-trades/-hours)
LOOSE = (3, 2)                         # reported as a sensitivity: contradicted with 3 trades over 2 hours
DEX_ONLY_HOURS = 2                     # robust hours beyond the threshold a DEX-only episode needs in its first day
SHIFTS_DAYS = (-28, -21, -14, -7, 7, 14, 21, 28)
STATUSES = ['confirmed', 'partial', 'contradicted', 'unverified']


def dex_series(assets, hourly, rates, rules, min_hour_trades):
    """symbol -> (mid deviation at traded hours, high deviation at hours with enough trades, trades per hour).
    hourly: one row per asset-hour (make_labels.collapse_dex_quotes)."""
    base = hourly[['symbol', 'hour', 'unit']].copy()
    base['source'] = 'dex'
    base['ts'] = np.nan
    base['hour'] = pd.to_datetime(base['hour'], utc=True).dt.floor('h')
    trades = hourly.assign(hour=base['hour'])
    have = set(hourly['symbol'])
    out = {}
    for a in assets:
        if a['symbol'] not in have:
            continue
        devs = []
        for col in ('price', 'high'):
            prices = base.assign(price=pd.to_numeric(hourly[col], errors='coerce')).dropna(subset=['price'])
            devs.append(ml.deviation(a, prices, rates, rules, 'dex')[0])
        mid, high = devs
        if mid.empty:
            continue
        n = trades[trades['symbol'] == a['symbol']].groupby('hour')['n_trades'].sum()
        traded = n.index[n > 0]
        robust = n.index[n >= min_hour_trades]
        out[a['symbol']] = (mid[mid.index.isin(traded)].dropna(), high[high.index.isin(robust)].dropna(), n)
    return out


def window(e):
    return e['start'] - BEFORE_H * H, min(e['end'], e['start'] + CHECK_H * H) + AFTER_H * H


def check_episode(e, theta, band, dex, min_trades=MIN_TRADES, min_hours=MIN_HOURS):
    empty = {'status': 'unverified', 'dex_hours': 0, 'dex_robust_hours': 0, 'dex_trades': 0,
             'dex_min_mid': np.nan, 'dex_max_mid': np.nan, 'dex_min_high': np.nan}
    if dex is None:
        return empty
    mid, high, n = dex
    w0, w1 = window(e)
    m, h = mid.loc[w0:w1], high.loc[w0:w1]
    ntr = int(n.loc[w0:w1].sum())
    hmin = float(h.min()) if len(h) else np.nan
    if len(h) and hmin <= -theta:
        status = 'confirmed'
    elif len(h) and hmin <= -band:
        status = 'partial'
    elif ntr >= min_trades and len(m) >= min_hours and float(m.abs().max()) < band:
        status = 'contradicted'         # traded at the reference all along (a far-off print is no evidence of a peg)
    else:
        status = 'unverified'
    return {'status': status, 'dex_hours': int(len(m)), 'dex_robust_hours': int(len(h)), 'dex_trades': ntr,
            'dex_min_mid': float(m.min()) if len(m) else np.nan, 'dex_max_mid': float(m.max()) if len(m) else np.nan,
            'dex_min_high': hmin}


def placebo(clean, all_eps, dex, rules, meta, evidence=(MIN_TRADES, MIN_HOURS)):
    """The check at shifted windows of the same asset where no labelled episode lies within a day."""
    by_sym = {s: g for s, g in all_eps.groupby('symbol')}
    rows = []
    for e in clean.itertuples(index=False):
        e = e._asdict()
        d = dex.get(e['symbol'])
        if d is None or not len(d[0]):
            continue
        first, last = d[0].index.min(), d[0].index.max()
        theta = rules['threshold'][meta[e['symbol']]['peg']]
        g = by_sym[e['symbol']]
        for k in SHIFTS_DAYS:
            s = {'start': e['start'] + k * D, 'end': e['end'] + k * D}
            w0, w1 = window(s)
            if w0 < first or w1 > last:
                continue
            if ((g['start'] - D <= w1) & (g['end'] + D >= w0)).any():
                continue
            rows.append({'symbol': e['symbol'], 'category': e['category'], 'episode_start': e['start'],
                         'shift_days': k, **check_episode(s, theta, theta * rules['recovery_band_frac'], d,
                                                          *evidence)})
    return pd.DataFrame(rows)


def dex_only(assets, dex, eps, rules):
    rows = []
    by_sym = {s: g for s, g in eps.groupby('symbol')}
    for a in assets:
        if a['symbol'] not in dex:
            continue
        mid, high, n = dex[a['symbol']]
        theta = rules['threshold'][a['peg']]
        found = ml.detect(high, theta, rules['min_consecutive_hours'], rules['single_hour_severe'],
                          rules['recovery_band_frac'], rules['recovery_hours'])
        g = by_sym.get(a['symbol'])
        for d in found:
            first = high.loc[d['start']:d['start'] + 24 * H]
            if (first <= -theta).sum() < DEX_ONLY_HOURS:
                continue
            if g is not None and ((g['start'] - D <= d['start']) & (g['end'] + D >= d['start'])).any():
                continue
            rows.append({'symbol': a['symbol'], 'category': a['category'], 'start': d['start'], 'end': d['end'],
                         'dex_min_high': round(float(high.loc[d['start']:d['end']].min()), 5),
                         'dex_robust_hours': int(len(high.loc[d['start']:d['end']])),
                         'dex_trades': int(n.loc[d['start']:d['end']].sum())})
    return pd.DataFrame(rows, columns=['symbol', 'category', 'start', 'end', 'dex_min_high', 'dex_robust_hours', 'dex_trades'])


def known_events(path, assets, dex, rules):
    peg = {a['symbol']: a['peg'] for a in assets}
    rows = []
    for k in json.loads(Path(path).read_text()):
        day = pd.Timestamp(k['date'], tz='UTC')
        d = dex.get(k['symbol'])
        m = d[0].loc[day - 72 * H:day + 72 * H] if d is not None else pd.Series(dtype=float)
        h = d[1].loc[day - 72 * H:day + 72 * H] if d is not None else pd.Series(dtype=float)
        theta = rules['threshold'].get(peg.get(k['symbol'], 'USD'), 0.01)
        rows.append({'symbol': k['symbol'], 'date': k['date'], 'expect': k['expect'], 'dex_hours': int(len(m)),
                     'dex_min_mid': round(float(m.min()), 4) if len(m) else np.nan,
                     'dex_min_high': round(float(h.min()), 4) if len(h) else np.nan,
                     'dex_beyond_threshold': bool(len(h) and h.min() <= -theta)})
    return pd.DataFrame(rows)


def table(df, by):
    t = pd.crosstab(df[by], df['status']).reindex(columns=STATUSES, fill_value=0)
    t['episodes'] = t.sum(axis=1)
    checked = t['confirmed'] + t['partial'] + t['contradicted']
    t['confirmed of checked'] = (t['confirmed'] / checked.where(checked > 0)).round(2)
    return t


def run(args):
    rules = json.loads(Path(args.rules).read_text())
    _, assets = load_registry(args.registry)
    targets = [a for a in assets if a['in_scope']]
    meta = {a['symbol']: a for a in targets}
    labels = Path(args.labels)
    eps = pd.read_csv(labels / 'episodes.csv')
    eps['suspect'] = eps['suspect'].fillna('').astype(str)
    eps['start'] = pd.to_datetime(eps['start'], utc=True)
    eps['end'] = pd.to_datetime(eps['end'], utc=True)
    raw = pd.read_csv(args.dex)
    if 'n_trades' not in raw:
        raw['n_trades'] = 1
    hourly = ml.collapse_dex_quotes(raw)
    rates = ml.load_rates(args.rates)
    args.contra_trades = getattr(args, 'contra_trades', MIN_TRADES)
    args.contra_hours = getattr(args, 'contra_hours', MIN_HOURS)
    dex = dex_series(targets, hourly, rates, rules, getattr(args, 'min_hour_trades', 2))

    rows = []
    for e in eps.itertuples(index=False):
        e = e._asdict()
        a = meta.get(e['symbol'])
        if a is None:
            continue
        theta = rules['threshold'][a['peg']]
        band = theta * rules['recovery_band_frac']
        loose = check_episode(e, theta, band, dex.get(e['symbol']), *LOOSE)['status']
        rows.append({**{k: e[k] for k in ('symbol', 'category', 'start', 'end', 'severity', 'held_min_dev', 'suspect')},
                     **check_episode(e, theta, band, dex.get(e['symbol']), args.contra_trades, args.contra_hours),
                     'status_loose': loose})
    chk = pd.DataFrame(rows)
    clean, flagged = chk[chk['suspect'] == ''], chk[chk['suspect'] != '']
    plc = placebo(eps[eps['suspect'] == ''][['symbol', 'category', 'start', 'end']], eps, dex, rules, meta,
                  (args.contra_trades, args.contra_hours))
    only = dex_only(targets, dex, eps, rules)
    kn = known_events(args.known, targets, dex, rules)
    chk.to_csv(labels / 'source_check.csv', index=False, float_format='%.5g')
    plc.to_csv(labels / 'source_check_placebo.csv', index=False, float_format='%.5g')
    only.to_csv(labels / 'source_check_dex_only.csv', index=False)
    excl = chk[(chk['status'] == 'contradicted') & (chk['suspect'] == '')][['symbol', 'start']]
    excl.to_csv(labels / 'exclude_contradicted.csv', index=False)

    top = clean['symbol'].value_counts().head(args.top).index
    sev = clean[clean['severity'].isin(['major', 'collapse'])]
    cov = pd.DataFrame([{'symbol': s, 'traded_hours': len(v[0]), 'robust_hours': len(v[1]), 'trades': int(v[2].sum())}
                        for s, v in dex.items()])
    both = clean.assign(sample='labelled episodes')
    if len(plc):
        both = pd.concat([both, plc.assign(sample='placebo windows')], ignore_index=True)
    L = ['# Episodes checked against DEX trade prices', '',
         f'Label source: `{shown(labels)}`; DEX prices: `{shown(args.dex)}`; reference values '
         f"{'with on-chain exchange rates' if rates is not None else 'without exchange rates (trailing medians)'}. "
         f'Window: {BEFORE_H} h before the start to {CHECK_H} h into the episode (or its end) plus {AFTER_H} h. '
         f'Confirmed / partial: an hour with at least {getattr(args, "min_hour_trades", 2)} trades whose upper-quartile price is at or '
         f'beyond the threshold / half the threshold. Contradicted: at least {args.contra_trades} trades over at least '
         f'{args.contra_hours} hours, with the median price within half the threshold of the reference in every traded hour. '
         f"With a looser standard ({LOOSE[0]} trades over {LOOSE[1]} hours), {int((clean['status_loose'] == 'contradicted').sum())} "
         'episodes would count as contradicted.', '',
         f"DEX prices for {len(dex)} of {len(targets)} assets (median {int(cov['traded_hours'].median()) if len(cov) else 0} "
         f"traded hours per asset, {int(cov['robust_hours'].median()) if len(cov) else 0} with at least "
         f'{getattr(args, "min_hour_trades", 2)} trades).', '',
         f'## All episodes that passed the quality filters ({len(clean)}), and placebo windows', '',
         'Placebo windows: the same asset and window length, shifted by '
         + ', '.join(f'{k:+d}' for k in SHIFTS_DAYS) + ' days, with no labelled episode within a day.', '',
         ml.md(table(both, 'sample')), '',
         '## By category', '', ml.md(table(clean, 'category')), '',
         f'## The {args.top} assets with the most episodes', '', ml.md(table(clean[clean['symbol'].isin(top)], 'symbol')), '',
         f'## Severe episodes (held depth 5% or more, {len(sev)})', '',
         ml.md(table(sev.assign(all='severe'), 'all')) if len(sev) else 'none', '',
         f'## Episodes flagged by the quality filters ({len(flagged)})', '',
         ml.md(flagged[['symbol', 'start', 'suspect', 'held_min_dev', 'status', 'dex_trades', 'dex_min_mid', 'dex_min_high']].round(4),
               index=False) if len(flagged) else 'none', '',
         f'## DEX-only episodes ({len(only)})', '',
         'Episodes of the robust DEX series (upper-quartile price, hours with enough trades), with at least two hours '
         'beyond the threshold in their first 24 hours, and no labelled episode of the asset within a day.', '',
         ml.md(only.groupby('symbol').size().sort_values(ascending=False).to_frame('episodes')) if len(only) else 'none', '',
         '## Known events', '', ml.md(kn, index=False) if len(kn) else 'none', '',
         f'{len(excl)} contradicted episodes written to `exclude_contradicted.csv` for a robustness run.', '']
    (labels / 'source_check.md').write_text('\n'.join(L))
    print('\n'.join(L[:14]))
    return chk, only, kn, plc


def shown(path):
    """A path as the reports print it: relative to the repository when inside it, so no machine path is written."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=str(ROOT / 'data' / 'labels'))
    ap.add_argument('--dex', default=str(ROOT / 'data' / 'dex_prices_hourly.csv'))
    ap.add_argument('--rates', default='')
    ap.add_argument('--registry', default=str(ROOT / 'config' / 'assets.json'))
    ap.add_argument('--rules', default=str(ROOT / 'config' / 'label_rules.json'))
    ap.add_argument('--known', default=str(ROOT / 'config' / 'known_events.json'))
    ap.add_argument('--min-hour-trades', type=int, default=2, help='trades an hour needs to confirm a depeg')
    ap.add_argument('--contra-trades', type=int, default=MIN_TRADES, help='trades a contradiction needs in the window')
    ap.add_argument('--contra-hours', type=int, default=MIN_HOURS, help='traded hours a contradiction needs in the window')
    ap.add_argument('--top', type=int, default=10)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
