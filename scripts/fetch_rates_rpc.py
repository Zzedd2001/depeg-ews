#!/usr/bin/env python3
"""Exchange rates (fair values) of LSTs, LRTs and yield-bearing wrappers via JSON-RPC.

Each grid time is resolved to the last block at or before it, then ONE eth_call
to Multicall3.aggregate3 reads every rate contract listed in config/assets.json
(Balancer-reviewed getRate() providers, or ERC-4626 convertToAssets(1e18)).
Rates move slowly (most update about daily), so a 6-hour grid is enough; the
label script forward-fills them, which never looks ahead.

Usage:
  export ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/<key>   # archive access needed
  python scripts/fetch_rates_rpc.py --check            # latest block only: sanity-check every rate spec
  python scripts/fetch_rates_rpc.py --decimals         # write config/decimals.json (needed by make_sql.py)
  python scripts/fetch_rates_rpc.py --start 2023-01-01 --end 2026-09-30 --step-hours 6
      (an interrupted run continues where it stopped when started again; --restart starts over)
  python scripts/fetch_rates_rpc.py --blocks data/hourly_blocks.csv   # use BigQuery hour->block map instead

Cost on Alchemy's free tier: one eth_call per grid time (~5,500 at 6 h) plus
about 4 block lookups each; far below the 30M CU monthly allowance.
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, check_rpc_url, http_json, iso_hour, load_registry, to_ts  # noqa: E402

AGG3 = bytes.fromhex('82ad56cb')        # aggregate3((address,bool,bytes)[])
DECIMALS = bytes.fromhex('313ce567')    # decimals()
SLOT = 12                               # post-merge slot time in seconds
RETRY_WAITS = (30, 60, 120, 300, 600)   # seconds to wait after a network failure before trying again


def patient(fn):
    """Call fn(), and when the connection fails (proxy hiccup, dropped tunnel, timeout) wait and try again
    (about 18 minutes in all) instead of stopping. HTTP errors that will not go away (401, 403, ...) and other
    exceptions are raised at once. Messages never show the endpoint URL, which holds the API key."""
    for k in range(len(RETRY_WAITS) + 1):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise
            if k == len(RETRY_WAITS):
                raise
            print(f'  network error ({getattr(e, "reason", e)}); trying again in {RETRY_WAITS[k]} s', flush=True)
            time.sleep(RETRY_WAITS[k])


def u256(x):
    return int(x).to_bytes(32, 'big')


def encode_aggregate3(calls):
    """calls: list of (target_hex, allow_failure, calldata_bytes) -> calldata bytes."""
    tails = []
    for target, allow, data in calls:
        pad = (-len(data)) % 32
        head = bytes(12) + bytes.fromhex(target[2:].lower()) + u256(1 if allow else 0) + u256(0x60)
        tails.append(head + u256(len(data)) + data + bytes(pad))
    offsets, cur = [], 32 * len(calls)
    for t in tails:
        offsets.append(cur)
        cur += len(t)
    return AGG3 + u256(0x20) + u256(len(calls)) + b''.join(u256(o) for o in offsets) + b''.join(tails)


def decode_aggregate3(ret):
    """returns list of (success, return_bytes)."""
    word = lambda b, i: int.from_bytes(b[i:i + 32], 'big')
    arr = word(ret, 0)
    n = word(ret, arr)
    base = arr + 32
    out = []
    for i in range(n):
        p = base + word(ret, base + 32 * i)
        ok = word(ret, p) != 0
        q = p + word(ret, p + 32)
        ln = word(ret, q)
        out.append((ok, ret[q + 32:q + 32 + ln]))
    return out


def rate_calldata(spec):
    sel = bytes.fromhex(spec['selector'][2:])
    return sel + (u256(10 ** 18) if spec.get('arg') == '1e18' else b'')


class Rpc:
    def __init__(self, url, sleep=0.0):
        self.url, self.sleep, self.id, self.ts_cache = url, sleep, 0, {}

    def batch(self, calls):
        payload = []
        for method, params in calls:
            self.id += 1
            payload.append({'jsonrpc': '2.0', 'id': self.id, 'method': method, 'params': params})
        res = http_json(self.url, payload)
        if isinstance(res, dict):  # some providers answer a single error object
            raise RuntimeError(res)
        res = sorted(res, key=lambda r: r['id'])
        time.sleep(self.sleep)
        return [r.get('result') if 'error' not in r else {'error': r['error']} for r in res]

    def block_ts(self, b):
        if b not in self.ts_cache:
            blk = self.batch([('eth_getBlockByNumber', [hex(b), False])])[0]
            self.ts_cache[b] = int(blk['timestamp'], 16)
        return self.ts_cache[b]

    def latest(self):
        b = int(self.batch([('eth_blockNumber', [])])[0], 16)
        self.head = b
        return b, self.block_ts(b)

    def block_at(self, t, hint):
        """Last block with timestamp <= t (hint = a nearby block number).

        Steps by the slot-time estimate; missed slots make each step overshoot a little,
        so the step is damped and the search falls back to bisection if it oscillates."""
        head = getattr(self, 'head', None) or self.latest()[0]
        clamp = lambda x: min(max(x, 0), head - 1)
        b = clamp(hint)
        lo, hi = 0, head          # invariant: answer in [lo, hi)
        for _ in range(64):
            ts = self.block_ts(b)
            if ts > t:
                hi = min(hi, b)
                nb = b - max(1, (ts - t + SLOT - 1) // SLOT)
            elif self.block_ts(b + 1) <= t:
                lo = max(lo, b + 1)
                nb = b + max(1, (t - ts) // SLOT)
            else:
                return b
            b = nb if lo <= nb < hi else (lo + hi) // 2
        raise RuntimeError(f'block search did not converge for t={t}')


def eth_call(rpc, block, calls):
    data = '0x' + encode_aggregate3(calls).hex()
    res = rpc.batch([('eth_call', [{'to': MULTICALL3, 'data': data}, hex(block)])])[0]
    if isinstance(res, dict) or res is None:
        return None
    return decode_aggregate3(bytes.fromhex(res[2:]))


def main():
    global MULTICALL3
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=str(ROOT / 'config' / 'assets.json'))
    ap.add_argument('--rpc', default=os.environ.get('ETH_RPC_URL', ''))
    ap.add_argument('--start', default='2023-01-01')
    ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--step-hours', type=int, default=6)
    ap.add_argument('--blocks', default='', help='CSV with columns hour,block_number (from sql/hourly_blocks.sql)')
    ap.add_argument('--sleep', type=float, default=0.05)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'rates.csv.gz'))
    ap.add_argument('--check', action='store_true', help='read every rate at the latest block and print it')
    ap.add_argument('--decimals', action='store_true', help='read decimals() of every registry token')
    ap.add_argument('--restart', action='store_true', help='start over instead of continuing an earlier run')
    args = ap.parse_args()
    check_rpc_url(args.rpc)

    doc, assets = load_registry(args.registry)
    MULTICALL3 = doc['multicall3']
    rpc = Rpc(args.rpc, args.sleep)
    latest_b, latest_ts = patient(rpc.latest)

    if args.decimals:
        calls = [(a['address'], True, DECIMALS) for a in assets]
        res = patient(lambda: eth_call(rpc, latest_b, calls))
        out = {a['symbol']: (int.from_bytes(r, 'big') if ok and len(r) == 32 else None) for a, (ok, r) in zip(assets, res)}
        (ROOT / 'config' / 'decimals.json').write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return

    rated = [a for a in assets if a.get('rate')]
    calls = [(a['rate']['contract'], True, rate_calldata(a['rate'])) for a in rated]

    if args.check:
        res = patient(lambda: eth_call(rpc, latest_b, calls))
        print(f'block {latest_b} ({iso_hour(latest_ts)})')
        for a, (ok, r) in zip(rated, res):
            val = int.from_bytes(r, 'big') / 1e18 if ok and len(r) == 32 else None
            flag = 'OK' if val is not None and 0.5 < val < 3 else 'CHECK'
            print(f"{a['symbol']:8s} {flag:5s} {val}  ({a['rate']['contract']})")
        return

    grid = list(range(to_ts(args.start), to_ts(args.end) + 24 * 3600, args.step_hours * 3600))
    block_map = {}
    if args.blocks:
        with open(args.blocks) as fh:
            for r in csv.DictReader(fh):
                block_map[to_ts(r['hour'][:13].replace(' ', 'T') + ':00')] = int(r['block_number'])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, resume = ([], None) if args.restart else previous_rows(out)
    if resume is not None:
        print(f'continuing an earlier run from {iso_hour(resume)} ({len(kept)} rows kept; --restart starts over)')
    todo = [t for t in grid if resume is None or t >= resume]
    hint = (int(kept[-1][3]) + args.step_hours * 3600 // SLOT) if kept else latest_b - (latest_ts - todo[0]) // SLOT if todo else latest_b
    n_ok = sum(r[5] == '1' for r in kept)
    with gzip.open(out, 'wt', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['symbol', 'time', 'ts', 'block', 'rate', 'ok'])
        w.writerows(kept)
        for i, t in enumerate(todo):
            if t > latest_ts:
                break
            def read(t=t, hint=hint):
                b = block_map.get(t) - 1 if t in block_map else rpc.block_at(t, hint)  # first block of hour - 1 = state at t
                return b, eth_call(rpc, b, calls)
            try:
                b, res = patient(read)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                sys.exit(f'The connection keeps failing ({getattr(e, "reason", e)}). Check the network or proxy, then run '
                         f'the same command again: it continues from {iso_hour(t)}.')
            hint = b + args.step_hours * 3600 // SLOT
            res = res or [(False, b'')] * len(calls)
            for a, (ok, r) in zip(rated, res):
                val = int.from_bytes(r, 'big') / 1e18 if ok and len(r) == 32 else ''
                w.writerow([a['symbol'], iso_hour(t), t, b, val, int(val != '')])
                n_ok += val != ''
            if i % 200 == 0:
                print(f'{iso_hour(t)} block {b}: {sum(1 for ok, r in res if ok and len(r) == 32)}/{len(calls)} rates '
                      f'({i + 1}/{len(todo)} grid times)', flush=True)
    print(f'wrote {out} ({n_ok} rate readings)')



def previous_rows(path):
    """Rows of an earlier run (a run that was killed may have left a truncated gzip). The last grid time
    may be incomplete, so its rows are dropped and the run continues from it. Returns (rows, resume ts)."""
    if not Path(path).exists():
        return [], None
    rows = []
    try:
        with gzip.open(path, 'rt', newline='') as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if len(row) == 6:
                    rows.append(row)
    except (OSError, EOFError, csv.Error):
        pass                                   # keep what could be read
    if not rows:
        return [], None
    last = max(int(row[2]) for row in rows)
    return [row for row in rows if int(row[2]) < last], last


MULTICALL3 = '0xcA11bde05977b3631167028862bE2a173976CA11'

if __name__ == '__main__':
    main()
