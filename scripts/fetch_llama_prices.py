#!/usr/bin/env python3
"""Hourly prices for every registry asset from the free DefiLlama coins API.

Quick path for the week-2 checkpoint (episode counts). DefiLlama aggregates
several venues and prices some derivative tokens from their exchange rate, so
these series are a first pass; paper-grade labels use DEX trade prices
(sql/ + scripts/make_sql.py) as the primary source and this as the second.

Usage:
  python scripts/fetch_llama_prices.py --start 2023-01-01 --end 2026-09-30
  python scripts/fetch_llama_prices.py --only USDe,ezETH --dry-run

Endpoint (checked 2026-10-02):
  https://coins.llama.fi/chart/{chain:address}?start=<unix>&span=<n>&period=1h&searchWidth=<s>
  -> {"coins": {key: {"symbol", "confidence", "decimals", "prices": [{"timestamp", "price"}]}}}
Without searchWidth the API only looks a few minutes around each hour and returns
nothing for many hours (e.g. rETH in Aug-Sep 2026); 1800 s fills those hours.
Assets can list fallback keys (registry field "llama_fallback", e.g. a coingecko:
id); they only fill hours the primary key leaves empty.
"""
import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, http_json, iso_hour, load_registry, round_to_hour, to_ts  # noqa: E402

BASE = 'https://coins.llama.fi'


def safe(key):
    return key.replace(':', '_').replace('/', '_')


def first_timestamps(keys, sleep):
    out = {}
    for i in range(0, len(keys), 10):
        chunk = keys[i:i + 10]
        res = http_json(f"{BASE}/prices/first/{','.join(chunk)}")
        coins = res.get('coins', res)
        for k in chunk:
            hit = coins.get(k) or coins.get(k.lower())
            out[k] = int(hit['timestamp']) if hit and 'timestamp' in hit else None
        time.sleep(sleep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=str(ROOT / 'config' / 'assets.json'))
    ap.add_argument('--start', default='2023-01-01')
    ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--span', type=int, default=500, help='hourly points per request')
    ap.add_argument('--search-width', type=int, default=1800,
                    help='seconds on either side of each hour in which DefiLlama looks for a price')
    ap.add_argument('--sleep', type=float, default=0.25)
    ap.add_argument('--only', default='', help='comma-separated symbols')
    ap.add_argument('--cache', default=str(ROOT / 'data' / 'raw' / 'llama'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'prices_llama_hourly.csv.gz'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    _, assets = load_registry(args.registry, in_scope_only=True)
    if args.only:
        wanted = set(args.only.split(',')) | {'WETH'}
        assets = [a for a in assets if a['symbol'] in wanted]
    start, end = to_ts(args.start), to_ts(args.end) + 23 * 3600
    chains = {a['symbol']: [a['llama_key']] + list(a.get('llama_fallback', [])) for a in assets}
    keys = list(dict.fromkeys(k for ks in chains.values() for k in ks))
    step, sw = args.span * 3600, args.search_width

    if args.dry_run:
        n = len(keys) * ((end - start) // step + 1)
        print(f'{len(assets)} assets, {len(keys)} keys, up to {n} requests (fewer after /prices/first trims each start)')
        print(f"example: {BASE}/chart/{keys[0]}?start={start}&span={args.span}&period=1h&searchWidth={sw}")
        return

    firsts = first_timestamps(keys, args.sleep)
    cache = Path(args.cache)
    rows, meta = {}, []
    for a in assets:
        sym = a['symbol']
        conf = dec = ''
        n_by_key = {}
        for rank, key in enumerate(chains[sym]):
            f = firsts.get(key)
            n_by_key[key] = 0
            if f is None:
                print(f'[skip] {sym}: no DefiLlama price history for {key}')
                continue
            t = max(start, (f // 3600) * 3600)
            d = cache / safe(key)
            d.mkdir(parents=True, exist_ok=True)
            while t <= end:
                fp = d / f'{t}_sw{sw}.json'
                if fp.exists():
                    res = json.loads(fp.read_text())
                else:
                    res = http_json(f'{BASE}/chart/{key}?start={t}&span={args.span}&period=1h&searchWidth={sw}')
                    fp.write_text(json.dumps(res))
                    time.sleep(args.sleep)
                coin = res.get('coins', {}).get(key) or next(iter(res.get('coins', {}).values()), None)
                if coin:
                    if rank == 0 or not conf:
                        conf, dec = coin.get('confidence', conf), coin.get('decimals', dec)
                    for p in coin.get('prices', []):
                        h = round_to_hour(p['timestamp'], tolerance=sw)
                        if h is None or h < start or h > end:
                            continue
                        prev = rows.get((sym, h))
                        if prev is not None and prev['llama_key'] != key:
                            continue  # a preferred key already covers this hour
                        if prev is None or abs(p['timestamp'] - h) < abs(prev['ts'] - h):
                            n_by_key[key] += prev is None
                            rows[(sym, h)] = {'symbol': sym, 'llama_key': key, 'hour': iso_hour(h), 'ts': p['timestamp'],
                                              'price': p['price'], 'confidence': coin.get('confidence', ''), 'source': 'defillama'}
                t += step
        n = sum(n_by_key.values())
        extra = sum(v for k, v in n_by_key.items() if k != chains[sym][0])
        print(f'{sym:8s} {n:6d} hourly prices' + (f' ({extra} from fallback keys)' if extra else ''))
        meta.append({'symbol': sym, 'llama_key': chains[sym][0], 'fallback_keys': ';'.join(chains[sym][1:]),
                     'first_ts': firsts.get(chains[sym][0]) or '', 'decimals': dec, 'confidence': conf,
                     'n_hours': n, 'n_hours_fallback': extra})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, 'wt', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['symbol', 'llama_key', 'hour', 'ts', 'price', 'confidence', 'source'])
        w.writeheader()
        for k in sorted(rows, key=lambda x: (x[0], x[1])):
            w.writerow(rows[k])
    with open(out.parent / 'token_meta_llama.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['symbol', 'llama_key', 'fallback_keys', 'first_ts', 'decimals', 'confidence',
                                           'n_hours', 'n_hours_fallback'])
        w.writeheader()
        w.writerows(meta)
    print(f'wrote {len(rows)} rows to {out}')


if __name__ == '__main__':
    main()
