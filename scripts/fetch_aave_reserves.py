#!/usr/bin/env python3
"""Aave v3 (core, Lido and EtherFi instances) and SparkLend reserves through an archive RPC.

The reserve list of each pool in config/lending_pools.json is read once at the latest
block. Then, at every grid time (default every 6 hours), ONE Multicall3 call per pool reads,
for every reserve:
  aToken.totalSupply()                       supplied amount (token units, interest included)
  variable / stable debt token totalSupply() borrowed amount
  Pool.getConfiguration(asset)               LTV, liquidation threshold and bonus, caps, frozen / paused
  Oracle.getAssetPrice(asset)                the price the pool itself uses (USD, 8 decimals)
  Oracle.getSourceOfAsset(asset)             the price-source contract (a change = an oracle swap)
and every e-mode category (LTV, threshold, bonus; from Aave 3.2 on also which reserves belong).

Usage:
  export ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/<key>     # archive access needed
  python scripts/fetch_aave_reserves.py --check        # latest block: print every reserve, test archive access
  python scripts/fetch_aave_reserves.py                # 2023-01-27 .. 2026-09-30, every 6 hours
Re-running continues after the last grid time already in the output (--restart starts over).

Outputs (data/lending/):
  aave_reserves.csv          pool, asset, symbol, time, block, supply, variable_debt, stable_debt,
                             price_usd, price_source, ltv, liq_threshold, liq_bonus, caps, flags, ...
  aave_emode.csv             pool, time, block, category, ltv, liq_threshold, liq_bonus, bitmaps, mode
  aave_reserves_meta.json    reserve list, token addresses, decimals, e-mode labels (latest block)
Cost on Alchemy's free tier: per grid time one JSON-RPC batch with one eth_call per pool, plus
about three block lookups; about 20,000 requests for 6-hourly points since 2023.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_rates_rpc as fr  # noqa: E402
from common import ROOT, check_rpc_url, iso_hour, load_registry, to_ts  # noqa: E402

SEL = {k: bytes.fromhex(v) for k, v in {
    'getReservesList': 'd1946dbc',                    # Pool
    'getReserveData': '35ea6a75',                     # Pool (15 static words; id = word 7, tokens = words 8-10)
    'getConfiguration': 'c44b11f7',                   # Pool
    'getEModeCategoryData': '6c6f6ae1',               # Pool (all versions; legacy struct with a label)
    'getEModeCategoryCollateralConfig': 'b286f467',   # Pool, Aave 3.2+
    'getEModeCategoryCollateralBitmap': 'b0771dba',   # Pool, Aave 3.2+
    'getEModeCategoryBorrowableBitmap': '903a2c71',   # Pool, Aave 3.2+
    'getEModeCategoryLabel': '2083e183',              # Pool, Aave 3.2+
    'getReserveTokensAddresses': 'd2493b6c',          # data provider (fallback for token addresses)
    'getAssetPrice': 'b3596f07',                      # oracle
    'getSourceOfAsset': '92bf2be0',                   # oracle
    'BASE_CURRENCY_UNIT': '8c89b64f',                 # oracle
    'totalSupply': '18160ddd', 'decimals': '313ce567', 'symbol': '95d89b41',   # ERC-20
}.items()}
ZERO = '0x' + '0' * 40
RESERVE_COLS = ['pool', 'asset', 'symbol', 'time', 'ts', 'block', 'supply', 'variable_debt', 'stable_debt',
                'price_usd', 'price_source', 'ltv', 'liq_threshold', 'liq_bonus', 'reserve_factor', 'active',
                'frozen', 'borrowing_enabled', 'paused', 'supply_cap', 'borrow_cap', 'debt_ceiling',
                'emode_legacy', 'reserve_id']
EMODE_COLS = ['pool', 'time', 'ts', 'block', 'category', 'ltv', 'liq_threshold', 'liq_bonus',
              'collateral_bitmap', 'borrowable_bitmap', 'mode']


# ---------- ABI helpers ----------
def call(target, name, *args):
    data = SEL[name]
    for a in args:
        data += bytes(12) + bytes.fromhex(a[2:]) if isinstance(a, str) else fr.u256(a)
    return (target, True, data)


def word(r, i):
    return int.from_bytes(r[32 * i:32 * i + 32], 'big')


def as_uint(ok_r):
    ok, r = ok_r
    return word(r, 0) if ok and len(r) >= 32 else None


def as_addr(ok_r):
    v = as_uint(ok_r)
    return None if v is None else '0x' + v.to_bytes(32, 'big')[12:].hex()


def as_addr_array(ok_r):
    ok, r = ok_r
    if not ok or len(r) < 64:
        return None
    off = word(r, 0)
    n = int.from_bytes(r[off:off + 32], 'big')
    return ['0x' + r[off + 32 + 32 * i + 12:off + 64 + 32 * i].hex() for i in range(n)]


def as_string(ok_r):
    """string or bytes32 return value (some old tokens use bytes32 symbols)."""
    ok, r = ok_r
    if not ok or len(r) < 32:
        return ''
    if len(r) == 32:
        return r.rstrip(b'\0').decode('utf-8', 'replace')
    off = word(r, 0)
    if off + 32 > len(r):
        return ''
    n = int.from_bytes(r[off:off + 32], 'big')
    return r[off + 32:off + 32 + n].decode('utf-8', 'replace')


def decode_config(c):
    """Aave v3 ReserveConfigurationMap bitmap -> dict (ratios as fractions, caps in whole tokens)."""
    bits = lambda lo, n: (c >> lo) & ((1 << n) - 1)
    bonus = bits(32, 16)
    return {'ltv': bits(0, 16) / 1e4, 'liq_threshold': bits(16, 16) / 1e4,
            'liq_bonus': round(bonus / 1e4 - 1, 6) if bonus else 0.0,
            'decimals': bits(48, 8), 'active': bits(56, 1), 'frozen': bits(57, 1),
            'borrowing_enabled': bits(58, 1), 'paused': bits(60, 1), 'reserve_factor': bits(64, 16) / 1e4,
            'borrow_cap': bits(80, 36), 'supply_cap': bits(116, 36), 'emode_legacy': bits(168, 8),
            'debt_ceiling': bits(212, 40) / 100}


def decode_emode_legacy(ok_r):
    """getEModeCategoryData -> (ltv, threshold, bonus, label); None if the call failed."""
    ok, r = ok_r
    if not ok or len(r) < 32 * 6:
        return None
    off = word(r, 0)
    ltv, lt, bonus = (int.from_bytes(r[off + 32 * i:off + 32 * i + 32], 'big') for i in range(3))
    label = ''
    lo = off + int.from_bytes(r[off + 128:off + 160], 'big')
    if lo + 32 <= len(r):
        n = int.from_bytes(r[lo:lo + 32], 'big')
        label = r[lo + 32:lo + 32 + n].decode('utf-8', 'replace')
    return ltv, lt, bonus, label


def ratio(bps, bonus=False):
    return (round(bps / 1e4 - 1, 6) if bps else 0.0) if bonus else bps / 1e4


# ---------- discovery at the latest block ----------
def discover(rpc, pool, block, registry_symbols, emode_max=0):
    """Reserve list, token addresses, decimals, symbols and e-mode ids of one pool."""
    res = fr.eth_call(rpc, block, [call(pool['pool'], 'getReservesList'), call(pool['oracle'], 'BASE_CURRENCY_UNIT')])
    if res is None:
        raise RuntimeError(f"{pool['name']}: eth_call failed at block {block}")
    assets = as_addr_array(res[0]) or []
    base_unit = as_uint(res[1]) or 10 ** 8
    calls = []
    for a in assets:
        calls += [call(pool['pool'], 'getReserveData', a), call(a, 'decimals'), call(a, 'symbol'),
                  call(pool['data_provider'], 'getReserveTokensAddresses', a)]
    res = fr.eth_call(rpc, block, calls) if calls else []
    reserves = []
    for i, a in enumerate(assets):
        rd, dec, sym, tok = res[4 * i:4 * i + 4]
        ok, r = rd
        if ok and len(r) >= 11 * 32:
            rid, a_tok, s_debt, v_debt = word(r, 7), *('0x' + r[32 * k + 12:32 * k + 32].hex() for k in (8, 9, 10))
        else:   # fall back to the data provider for token addresses; no reserve id
            ok2, r2 = tok
            if not ok2 or len(r2) < 96:
                print(f"  {pool['name']}: cannot read reserve data of {a}; skipped")
                continue
            rid, a_tok, s_debt, v_debt = None, *('0x' + r2[32 * k + 12:32 * k + 32].hex() for k in range(3))
        reserves.append({'asset': a.lower(), 'symbol': registry_symbols.get(a.lower()) or as_string(sym) or a.lower(),
                         'decimals': as_uint(dec) if as_uint(dec) is not None else 18, 'reserve_id': rid,
                         'a_token': a_tok, 'stable_debt': s_debt, 'variable_debt': v_debt})
    # e-mode ids: probe 1..255 with both getters (legacy struct; 3.2+ config and label), keep up to the highest id in use
    probe = range(1, (emode_max or 255) + 1)
    calls = []
    for c in probe:
        calls += [call(pool['pool'], 'getEModeCategoryData', c), call(pool['pool'], 'getEModeCategoryCollateralConfig', c),
                  call(pool['pool'], 'getEModeCategoryLabel', c)]
    res = fr.eth_call(rpc, block, calls) or []
    labels = {}
    for k, c in enumerate(probe):
        legacy, conf, label = res[3 * k:3 * k + 3] if len(res) >= 3 * k + 3 else (None, None, None)
        d = decode_emode_legacy(legacy) if legacy else None
        new = [word(conf[1], j) for j in range(3)] if conf and conf[0] and len(conf[1]) >= 96 else [0, 0, 0]
        name = (d[3] if d else '') or (as_string(label) if label else '')
        if (d and (d[0] or d[1])) or new[0] or new[1] or name:
            labels[c] = name
    n_emode = emode_max or (max(labels) if labels else 0)
    return {'name': pool['name'], 'base_unit': base_unit, 'reserves': reserves,
            'emode_ids': list(range(1, n_emode + 1)), 'emode_labels': labels}


def state_calls(pool, meta):
    """The per-grid-time call list of one pool (fixed order, decoded by read_state)."""
    calls = []
    for rv in meta['reserves']:
        a = rv['asset']
        calls += [call(rv['a_token'], 'totalSupply'), call(rv['variable_debt'], 'totalSupply'),
                  call(rv['stable_debt'], 'totalSupply') if rv['stable_debt'] != ZERO else call(a, 'decimals'),
                  call(pool['pool'], 'getConfiguration', a), call(pool['oracle'], 'getAssetPrice', a),
                  call(pool['oracle'], 'getSourceOfAsset', a)]
    for c in meta['emode_ids']:
        calls += [call(pool['pool'], 'getEModeCategoryData', c), call(pool['pool'], 'getEModeCategoryCollateralConfig', c),
                  call(pool['pool'], 'getEModeCategoryCollateralBitmap', c), call(pool['pool'], 'getEModeCategoryBorrowableBitmap', c)]
    return calls


def read_state(meta, res, t, block):
    """Decode one pool's multicall result -> (reserve rows, e-mode rows)."""
    rows, erows, n = [], [], 6 * len(meta['reserves'])
    for i, rv in enumerate(meta['reserves']):
        sup, vdebt, sdebt, conf, price, src = res[6 * i:6 * i + 6]
        c = as_uint(conf)
        if not c:      # reserve not listed yet at this block (or call failed)
            continue
        cfg = decode_config(c)
        scale = 10 ** rv['decimals']
        val = lambda x: '' if as_uint(x) is None else as_uint(x) / scale
        p = as_uint(price)
        rows.append({'pool': meta['name'], 'asset': rv['asset'], 'symbol': rv['symbol'], 'time': iso_hour(t), 'ts': t,
                     'block': block, 'supply': val(sup), 'variable_debt': val(vdebt),
                     'stable_debt': val(sdebt) if rv['stable_debt'] != ZERO else 0.0,
                     'price_usd': '' if p is None else p / meta['base_unit'], 'price_source': as_addr(src) or '',
                     **{k: cfg[k] for k in ('ltv', 'liq_threshold', 'liq_bonus', 'reserve_factor', 'active', 'frozen',
                                            'borrowing_enabled', 'paused', 'supply_cap', 'borrow_cap', 'debt_ceiling',
                                            'emode_legacy')},
                     'reserve_id': '' if rv['reserve_id'] is None else rv['reserve_id']})
    for j, cat in enumerate(meta['emode_ids']):
        legacy, conf32, cbm, bbm = res[n + 4 * j:n + 4 * j + 4]
        new = as_uint(conf32) is not None and len(conf32[1]) >= 96
        if new:   # Aave 3.2+: config struct and membership bitmaps
            ltv, lt, bonus = (word(conf32[1], k) for k in range(3))
        else:
            d = decode_emode_legacy(legacy)
            if not d:
                continue
            ltv, lt, bonus = d[:3]
        if not (ltv or lt):
            continue
        erows.append({'pool': meta['name'], 'time': iso_hour(t), 'ts': t, 'block': block, 'category': cat,
                      'ltv': ratio(ltv), 'liq_threshold': ratio(lt), 'liq_bonus': ratio(bonus, True),
                      'collateral_bitmap': hex(as_uint(cbm)) if new and as_uint(cbm) is not None else '',
                      'borrowable_bitmap': hex(as_uint(bbm)) if new and as_uint(bbm) is not None else '',
                      'mode': 'bitmap' if new else 'legacy'})
    return rows, erows


# ---------- output with resume ----------
def resume(paths, restart):
    """Return the grid time to resume from (or None) after dropping rows of the last, maybe partial, time."""
    if restart:
        for p in paths:
            p.unlink(missing_ok=True)
        return None
    if not paths[0].exists():
        return None
    with open(paths[0], newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    last = max(int(r['ts']) for r in rows if r.get('ts', '').isdigit())
    for p in paths:
        if not p.exists():
            continue
        with open(p, newline='') as fh:
            rd = csv.DictReader(fh)
            cols, keep = rd.fieldnames, [r for r in rd if r.get('ts', '').isdigit() and int(r['ts']) < last]
        with open(p, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(keep)
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pools', default=str(ROOT / 'config' / 'lending_pools.json'))
    ap.add_argument('--registry', default=str(ROOT / 'config' / 'assets.json'))
    ap.add_argument('--rpc', default=os.environ.get('ETH_RPC_URL', ''))
    ap.add_argument('--start', default='2023-01-27')
    ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--step-hours', type=int, default=6)
    ap.add_argument('--only', default='', help='comma-separated pool names (default: all in --pools)')
    ap.add_argument('--emode-max', type=int, default=0, help='highest e-mode id to read (default: found at the latest block)')
    ap.add_argument('--sleep', type=float, default=0.05)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'lending'))
    ap.add_argument('--check', action='store_true', help='latest block only: print every reserve and test archive access')
    ap.add_argument('--restart', action='store_true', help='discard earlier output and start over')
    args = ap.parse_args()
    check_rpc_url(args.rpc)

    doc, assets = load_registry(args.registry)
    fr.MULTICALL3 = doc['multicall3']
    reg = {a['address'].lower(): a['symbol'] for a in assets}
    pools = json.loads(Path(args.pools).read_text())['instances']
    if args.only:
        pools = [p for p in pools if p['name'] in args.only.split(',')]
    rpc = fr.Rpc(args.rpc, args.sleep)
    latest_b, latest_ts = rpc.latest()
    metas = [discover(rpc, p, latest_b, reg, args.emode_max) for p in pools]
    for p, m in zip(pools, metas):
        print(f"{p['name']}: {len(m['reserves'])} reserves, e-mode ids 1..{len(m['emode_ids'])}, base unit {m['base_unit']}")

    if args.check:
        print(f'block {latest_b} ({iso_hour(latest_ts)})')
        for p, m in zip(pools, metas):
            res = fr.eth_call(rpc, latest_b, state_calls(p, m))
            print(f"\n{p['name']}")
            if res is None:
                print('  eth_call failed (check the addresses in lending_pools.json and the RPC endpoint)')
                continue
            rows, erows = read_state(m, res, latest_ts, latest_b)
            print(f"  {'symbol':10s} {'supply':>16s} {'debt':>16s} {'price':>12s} {'ltv':>5s} {'lt':>5s} frz src")
            for r in rows:
                debt = (r['variable_debt'] or 0) + (r['stable_debt'] or 0)
                print(f"  {r['symbol'][:10]:10s} {r['supply'] or 0:16,.0f} {debt:16,.0f} {r['price_usd'] or 0:12,.4f} "
                      f"{r['ltv']:5.2f} {r['liq_threshold']:5.2f} {r['frozen']:3d} {r['price_source']}")
            for e in erows:
                print(f"  e-mode {e['category']:3d} ltv {e['ltv']:.3f} lt {e['liq_threshold']:.3f} ({e['mode']}) "
                      f"{m['emode_labels'].get(e['category'], '')}")
        back = latest_b - 365 * 7200
        ok = fr.eth_call(rpc, back, [call(pools[0]['pool'], 'getReservesList')])
        print(f"\narchive access at block {back} (about a year ago): {'OK' if ok and ok[0][0] else 'NOT AVAILABLE - use an archive endpoint'}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_out = {m['name']: {**m, 'emode_labels': {str(k): v for k, v in m['emode_labels'].items()}} for m in metas}
    (out / 'aave_reserves_meta.json').write_text(json.dumps({'block': latest_b, 'time': iso_hour(latest_ts), 'pools': meta_out}, indent=1))
    paths = [out / 'aave_reserves.csv', out / 'aave_emode.csv']
    first = resume(paths, args.restart)
    step = args.step_hours * 3600
    t0 = to_ts(args.start)
    grid = [t for t in range(t0, to_ts(args.end) + 86400, step) if t <= latest_ts and (first is None or t >= first)]
    if first is not None:
        print(f'resuming at {iso_hour(first)}')
    calls = {p['name']: fr.encode_aggregate3(state_calls(p, m)) for p, m in zip(pools, metas)}
    starts = {p['name']: to_ts(p.get('start', args.start)) for p in pools}
    new = [not p.exists() or p.stat().st_size == 0 for p in paths]
    fh_r, fh_e = open(paths[0], 'a', newline=''), open(paths[1], 'a', newline='')
    w_r, w_e = csv.DictWriter(fh_r, RESERVE_COLS), csv.DictWriter(fh_e, EMODE_COLS)
    if new[0]:
        w_r.writeheader()
    if new[1]:
        w_e.writeheader()
    hint = latest_b - (latest_ts - grid[0]) // fr.SLOT if grid else latest_b
    errors = 0
    for i, t in enumerate(grid):
        b = rpc.block_at(t, hint)
        hint = b + step // fr.SLOT
        live = [(p, m) for p, m in zip(pools, metas) if t >= starts[p['name']]]
        if not live:
            continue
        res = rpc.batch([('eth_call', [{'to': fr.MULTICALL3, 'data': '0x' + calls[p['name']].hex()}, hex(b)]) for p, _ in live])
        n_rows = 0
        for (p, m), r in zip(live, res):
            if not isinstance(r, str):
                errors += 1
                if errors <= 5:
                    print(f"  {iso_hour(t)} {p['name']}: eth_call failed: {str(r)[:200]}")
                continue
            rows, erows = read_state(m, fr.decode_aggregate3(bytes.fromhex(r[2:])), t, b)
            w_r.writerows(rows)
            w_e.writerows(erows)
            n_rows += len(rows)
        fh_r.flush()
        fh_e.flush()
        if i % 100 == 0:
            print(f'{iso_hour(t)} block {b}: {n_rows} reserve rows from {len(live)} pools')
    fh_r.close()
    fh_e.close()
    print(f'wrote {paths[0]} and {paths[1]}' + (f' ({errors} failed pool reads)' if errors else ''))


if __name__ == '__main__':
    main()
