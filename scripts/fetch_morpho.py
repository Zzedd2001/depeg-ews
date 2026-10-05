#!/usr/bin/env python3
"""Morpho Blue markets, oracles, market history and MetaMorpho vault allocations.

Free public API, no key: https://api.morpho.org/graphql. The documented limit is 750
requests per minute, and roughly 20,000 requests in an hour is treated as abuse and
blocked for days. This script waits 0.5 s between requests by default and stops,
instead of waiting, if the API asks for a pause longer than 15 minutes.

Usage:
  python scripts/fetch_morpho.py --interval DAY     # first pass: every market and vault, daily points
  python scripts/fetch_morpho.py --interval HOUR    # then hourly points (30-day requests), only for markets that ever held
                                                    # >= $1M and vaults that ever put >= $100k into them
                                                    # (decided from the DAY files; see --hour-min-usd)
  python scripts/fetch_morpho.py --interval DAY --limit 5   # a quick test run
  python scripts/fetch_morpho.py --reclassify       # recompute oracle classes in markets.csv only (no network)
Re-runs reuse cached responses (data/raw/morpho), so an interrupted run continues.

Optional: with ETH_RPC_URL set (any Ethereum RPC, archive not needed), the script also reads
description() of every price feed the oracles use, which tells exchange-rate feeds
("wstETH / stETH Exchange Rate") apart from market-price feeds. The Morpho API itself has no
feed descriptions. Without it those oracles are classed as "feed"; running again later with
ETH_RPC_URL set reclassifies them from the cached responses.

Outputs (data/morpho/):
  markets.csv                         markets whose collateral or loan asset is a registry asset,
                                      with LLTV, oracle feeds and an oracle class (see classify_oracle)
  feed_descriptions.json              price feed address -> description() (only with ETH_RPC_URL)
  market_history_<interval>.csv.gz    market_id, ts, field, value
                                      (supplyAssetsUsd, borrowAssetsUsd, collateralAssetsUsd in USD;
                                       collateralAssets in collateral-token units)
  vaults.csv                          vaults that lend into those markets or lend one of their loan assets
  vault_allocation_<interval>.csv.gz  vault, market_id, ts, field, value
                                      (supplyAssetsUsd per market; totalAssetsUsd with an empty market_id;
                                       '...Assets' fields, if the API has no USD history, are in token units)

Field names follow the Morpho API docs (2026-10). If the API rejects an optional field,
the script drops it (or switches to its token-unit twin), says so, and carries on.
Only MetaMorpho (V1) vaults are read; Morpho Vault V2 allocates through adapters and
is not covered.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_registry, to_ts  # noqa: E402

API = 'https://api.morpho.org/graphql'
ZERO = '0x0000000000000000000000000000000000000000'
MORPHO_START = '2024-01-01'   # Morpho Blue went live on Ethereum in January 2024
PAGE = 500
VALIDATION = re.compile(r'Cannot query field|Unknown argument|Unknown type|Syntax Error|Variable "\$|'
                        r'Expected type|must have a selection|is not defined|GRAPHQL_VALIDATION', re.I)
RATE_FEED = re.compile(r'exchange rate|fundamental|redemption|conversion|rate provider|\bnav\b', re.I)
REFERENCE = re.compile(r'^\s*(W?ETH|W?BTC|USDC|USDT)\s*/\s*(USD|W?ETH|W?BTC)\s*$', re.I)   # a reference asset's price
DESCRIPTION = bytes.fromhex('7284e416')            # description() of a Chainlink-style price feed
ROUND_CAPS = {100, 500, 1000, 2000, 5000, 10000}   # a timeseries this long may have been cut by a server-side cap
TOO_BIG = re.compile(r'complex|too (large|many|big)|exceed|maximum|limit', re.I)   # page too large for the API

ORACLE_TYPES = ('MorphoChainlinkOracleData', 'MorphoChainlinkOracleV2Data')
OBJ_FIELDS = {'baseFeedOne', 'baseFeedTwo', 'baseOracleVault', 'quoteFeedOne', 'quoteFeedTwo', 'quoteOracleVault'}
BASE_SIDE = {'baseFeedOne', 'baseFeedTwo', 'baseOracleVault'}
# GraphQL type -> fields the script can do without (dropped, with a note, if the API does not know them)
MARKET_OPT = {
    'Oracle': ['type'],
    'MorphoChainlinkOracleData': ['baseFeedOne', 'baseFeedTwo', 'baseOracleVault', 'quoteFeedOne', 'quoteFeedTwo',
                                  'scaleFactor', 'vaultConversionSample'],
    'MorphoChainlinkOracleV2Data': ['baseFeedOne', 'baseFeedTwo', 'baseOracleVault', 'baseVaultConversionSample',
                                    'quoteFeedOne', 'quoteFeedTwo', 'quoteOracleVault', 'quoteVaultConversionSample',
                                    'scaleFactor'],
    'MarketState': ['supplyAssetsUsd', 'borrowAssetsUsd', 'collateralAssetsUsd'],
}
VAULT_OPT = {'VaultState': ['totalAssetsUsd'], 'VaultAllocation': ['supplyAssetsUsd']}

Q_MARKETS = """query Markets($first: Int, $skip: Int) {
  markets(first: $first, skip: $skip, where: { chainId_in: [1] }) {
    items {
      marketId
      lltv
      loanAsset { address symbol decimals }
      collateralAsset { address symbol decimals }
      oracle { address %(oracle)s %(data)s }
      %(state)s
    }
    pageInfo { countTotal }
  }
}"""

Q_VAULTS = """query Vaults($first: Int, $skip: Int) {
  vaults(first: $first, skip: $skip, where: { chainId_in: [1] }) {
    items {
      address name symbol
      asset { address symbol decimals }
      state { %(total)s allocation { market { marketId } %(alloc)s } }
    }
    pageInfo { countTotal }
  }
}"""

Q_MARKET_HISTORY = """query H($id: String!, $chainId: Int!, $options: TimeseriesOptions) {
  marketById(marketId: $id, chainId: $chainId) {
    historicalState { %s }
  }
}"""

Q_VAULT_HISTORY = """query VA($address: String!, $chainId: Int!, $options: TimeseriesOptions) {
  vaultByAddress(address: $address, chainId: $chainId) {
    historicalState { %s }
  }
}"""

MARKET_FIELDS = ['supplyAssetsUsd', 'borrowAssetsUsd', 'collateralAssetsUsd', 'collateralAssets']
VAULT_FIELDS = ['supplyAssetsUsd', 'totalAssetsUsd']            # supplyAssets* sit inside allocation { }
TWINS = {'supplyAssetsUsd': 'supplyAssets', 'totalAssetsUsd': 'totalAssets'}   # vault USD field -> token-unit twin


def selection(fields):
    return ' '.join(f'{f} {{ address }}' if f in OBJ_FIELDS else f for f in fields)


def q_markets(opt):
    frags = ' '.join(f'... on {t} {{ {selection(opt[t])} }}' for t in ORACLE_TYPES if opt.get(t))
    return Q_MARKETS % {'oracle': ' '.join(opt.get('Oracle', [])),
                        'data': f'data {{ __typename {frags} }}' if frags else '',
                        'state': 'state { %s }' % ' '.join(opt['MarketState']) if opt.get('MarketState') else ''}


def q_vaults(opt):
    return Q_VAULTS % {'total': ' '.join(opt.get('VaultState', [])), 'alloc': ' '.join(opt.get('VaultAllocation', []))}


class GqlError(RuntimeError):
    pass


class Gql:
    def __init__(self, sleep, cache_dir, url=API):
        self.sleep, self.cache, self.url, self.n = sleep, Path(cache_dir), url, 0
        self.cache.mkdir(parents=True, exist_ok=True)
        self.warned = set()

    def post(self, payload):
        data = json.dumps(payload).encode()
        for attempt in range(6):
            req = urllib.request.Request(self.url, data=data, headers={
                'Content-Type': 'application/json', 'User-Agent': 'depeg-ews/0.4 (research)'})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = int(e.headers.get('Retry-After') or 60)
                    if wait > 900:
                        sys.exit(f'Morpho API asks to wait {wait} s (rate-limit ban). Stop and retry later; '
                                 f'cached responses are kept.')
                    print(f'  rate limited, waiting {wait} s')
                    time.sleep(wait)
                    continue
                raw = e.read().decode(errors='replace')
                if e.code in (500, 502, 503, 504) and attempt < 5:
                    time.sleep(5 * (attempt + 1))
                    continue
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                if isinstance(body, dict) and body.get('errors'):
                    return body        # GraphQL errors (validation failures come back as HTTP 400): query() decides
                raise GqlError(f'HTTP {e.code}: {raw[:500]}')
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt < 5:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        raise GqlError('too many retries')

    def query(self, q, variables):
        key = hashlib.sha1(json.dumps([q, variables], sort_keys=True).encode()).hexdigest()
        fp = self.cache / f'{key}.json.gz'
        if fp.exists():
            res = json.loads(gzip.decompress(fp.read_bytes()).decode())
        else:
            res = self.post({'query': q, 'variables': variables})
            self.n += 1
            time.sleep(self.sleep)
        errors = [e.get('message', str(e)) for e in res.get('errors') or []]
        if errors and (res.get('data') is None or any(VALIDATION.search(m) for m in errors)):
            raise GqlError('; '.join(errors))
        if not fp.exists():
            fp.write_bytes(gzip.compress(json.dumps(res).encode()))
        for m in errors:   # partial data, e.g. a USD value the API could not price: keep the rest
            short = m[:120]
            if short not in self.warned and len(self.warned) < 10:
                self.warned.add(short)
                print(f'  API warning (data kept, affected values empty): {short}')
        return res['data']


def unknown_fields(msg):
    """Every 'Cannot query field "x" on type "Y"' in an error text -> [(x, Y)] (quotes may be escaped)."""
    return list(dict.fromkeys(re.findall(r'Cannot query field \\?"(\w+)\\?" on type \\?"(\w+)\\?"', msg)))


def unknown_types(msg):
    return list(dict.fromkeys(re.findall(r'Unknown type \\?"(\w+)\\?"', msg)))


def drop_unknown(opt, msg):
    """Remove from opt (GraphQL type -> optional fields) everything the error says the API does not know.
    Returns False if the error is about anything else; the caller then re-raises it."""
    pairs, types = unknown_fields(msg), unknown_types(msg)
    if not pairs and not types:
        return False
    for t in types:
        if t not in opt:
            return False
        print(f'  API has no type "{t}"; continuing without it')
        opt[t] = []
    for f, t in pairs:
        group = t if f in opt.get(t, []) else next((g for g, fs in opt.items() if f in fs), None)
        if group is None:
            return False
        opt[group] = [x for x in opt[group] if x != f]
        print(f'  API has no field "{f}" on {t}; continuing without it')
    return True


def adapt(fields, bad, twins=None):
    """Drop an unknown field, or swap a USD field for its token-unit twin."""
    twin = (twins or {}).get(bad)
    if twin and twin not in fields:
        print(f'  API has no field "{bad}"; using "{twin}" (token units) instead')
        return [twin if f == bad else f for f in fields]
    print(f'  API has no field "{bad}"; continuing without it')
    return [f for f in fields if f != bad]


def history_query(gql, template, fields, variables, render, twins=None):
    """Run a timeseries query, adapting the field list to what the API knows."""
    while fields:
        try:
            return gql.query(template % render(fields), variables), fields
        except GqlError as e:
            bad = list(dict.fromkeys(f for f, _ in unknown_fields(str(e))))
            if not bad or any(f not in fields for f in bad):
                raise
            for f in bad:
                fields = adapt(fields, f, twins)
    return None, fields


def paginate(gql, make_query, root, opt=None):
    """All pages of a list query. make_query(opt) -> query text, where opt maps a GraphQL type to its
    optional fields; fields the API does not know are dropped and the listing starts over."""
    opt = {k: list(v) for k, v in (opt or {}).items()}
    page = PAGE
    while True:
        try:
            items, skip = [], 0
            while True:
                data = gql.query(make_query(opt), {'first': page, 'skip': skip})[root]
                batch = data.get('items') or []
                items += batch
                skip += len(batch)
                total = (data.get('pageInfo') or {}).get('countTotal')
                if not batch or (total is not None and skip >= total) or (total is None and len(batch) < page):
                    return items, opt
        except GqlError as e:
            if drop_unknown(opt, str(e)):
                continue
            if TOO_BIG.search(str(e)) and page >= 50:
                page //= 2
                print(f'  API refused the page size ({str(e)[:100]}); trying {page} items per page')
                continue
            raise


def addr(x):
    return ((x or {}).get('address') or '').lower()


def read_descriptions(rpc_url, feeds, path):
    """description() of each price-feed contract at the latest block (cached in `path`); cache only without an RPC."""
    known = json.loads(path.read_text()) if path.exists() else {}
    todo = sorted({f for f in feeds if f and f != ZERO and f not in known})
    if todo and rpc_url:
        import fetch_rates_rpc as fr
        from fetch_aave_reserves import as_string
        rpc = fr.Rpc(rpc_url)
        block, _ = rpc.latest()
        for i in range(0, len(todo), 200):
            part = todo[i:i + 200]
            res = fr.eth_call(rpc, block, [(f, True, DESCRIPTION) for f in part])
            if res is None:
                print('  eth_call for feed descriptions failed; exchange-rate feeds not detected this run')
                return known
            for f, r in zip(part, res):
                known[f] = as_string(r)
        path.write_text(json.dumps(known, indent=1, sort_keys=True))
    return known


def _plain(symbol):
    s = (symbol or '').strip().upper()
    return s[1:] if s in ('WETH', 'WBTC') else s


def base_class(texts, collateral_symbol='', vault=False):
    """Oracle class from the description() texts of the collateral-side price feeds (see classify_oracle).
    Each feed is an exchange-rate / fundamental feed, a reference-asset price (ETH, BTC, USDC or USDT in
    USD, ETH or BTC, unless the collateral is that very token) or a market price of something in the
    collateral's path. With no market price among them, the collateral is valued at its exchange rate (or
    1:1) in a reference asset, and a market depeg of the collateral does not show."""
    if not texts or not all(texts):
        return 'feed'                                  # a feed without a description: assume a market price
    rate = [bool(RATE_FEED.search(t)) for t in texts]
    ref = [bool(m) and _plain(m.group(1)) != _plain(collateral_symbol) for m in (REFERENCE.match(t) for t in texts)]
    if not all(a or b for a, b in zip(rate, ref)):
        return 'feed'
    return 'rate_feed' if any(rate) else ('vault_rate' if vault else 'fixed')


def classify_oracle(o, desc=None, available=None, collateral_symbol=''):
    """How the oracle prices the COLLATERAL (the 'base' side of a Morpho Chainlink oracle).

    none:       no oracle (idle market)
    custom:     not a standard Morpho Chainlink oracle, or its base side could not be read; inspect by hand
    fixed:      no base feed and no base vault (a constant price), or only reference-asset feeds (the
                collateral is valued 1:1 as ETH, BTC, USDC or USDT): blind to any depeg of the collateral
    vault_rate: base priced by an ERC-4626 conversion, alone or times reference-asset feeds
    rate_feed:  base feeds are exchange-rate / fundamental feeds, possibly times reference-asset feeds
                (e.g. "weETH/ETH exchange rate" x "ETH / USD"), judged from their on-chain description()
    feed:       at least one base feed that is (as far as its description shows) a market price of the
                collateral or of a token in its path (e.g. "STETH / USD" for wstETH)
    fixed, vault_rate and rate_feed mean a market depeg of the collateral does not trigger liquidations.
    desc: feed address -> description() text; available: GraphQL type -> the fields that were queried."""
    if not o or not o.get('address') or o['address'].lower() == ZERO:
        return 'none'
    d = o.get('data') or {}
    t = d.get('__typename')
    if not d or (t and t not in ORACLE_TYPES):
        return 'custom'
    if available is not None and t and not BASE_SIDE <= set(available.get(t, [])):
        return 'custom'
    feeds = [addr(d.get(k)) for k in ('baseFeedOne', 'baseFeedTwo')]
    feeds = [f for f in feeds if f and f != ZERO]
    vault = addr(d.get('baseOracleVault'))
    has_vault = bool(vault and vault != ZERO)
    if not feeds:
        return 'vault_rate' if has_vault else 'fixed'
    return base_class([(desc or {}).get(f, '') for f in feeds], collateral_symbol, has_vault)


def reclassify(out):
    """Recompute oracle_class in markets.csv from its saved feed columns and feed_descriptions.json, without
    the network (for a new classification rule; 'none' and 'custom' need the API and are kept)."""
    path = Path(out) / 'markets.csv'
    dpath = Path(out) / 'feed_descriptions.json'
    desc = {k.lower(): v for k, v in json.loads(dpath.read_text()).items()} if dpath.exists() else {}
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    changes, counts = {}, {}
    for r in rows:
        if r['oracle_class'] not in ('none', 'custom'):
            feeds = [f.lower() for f in (r.get('base_feed_1') or '', r.get('base_feed_2') or '') if f and f.lower() != ZERO]
            vault = (r.get('base_vault') or '').lower() not in ('', ZERO)
            new = (base_class([desc.get(f, '') for f in feeds], r.get('collateral_symbol', ''), vault) if feeds
                   else ('vault_rate' if vault else 'fixed'))
            if new != r['oracle_class']:
                key = f"{r['oracle_class']} -> {new}"
                changes[key] = changes.get(key, 0) + 1
                r['oracle_class'] = new
        counts[r['oracle_class']] = counts.get(r['oracle_class'], 0) + 1
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'{path}: {len(rows)} markets; changed: {changes or "none"}; oracle classes now: {counts}')
    return changes


def chunks(start, end, days):
    t = start
    while t < end:
        yield t, min(t + days * 86400, end)
        t += days * 86400


def write_gz(path, header, rows):
    with gzip.open(path, 'wt', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def day_maxima(path, key_col, fields, by_market=False):
    """Largest daily value per id (or per (vault, market)) in a DAY output file; {} if absent."""
    out = {}
    if not Path(path).exists():
        return out
    with gzip.open(path, 'rt') as fh:
        for r in csv.DictReader(fh):
            if r['field'] not in fields:
                continue
            k = (r[key_col], r['market_id']) if by_market else r[key_col]
            out[k] = max(out.get(k, 0.0), float(r['value']))
    return out


def active_windows(path, key_col, sel_ids=None):
    """(first, last) timestamp with a positive value per id in a DAY output file; for vault files only
    allocations to `sel_ids` count. {} if the file is absent."""
    out = {}
    if not Path(path).exists():
        return out
    with gzip.open(path, 'rt') as fh:
        for r in csv.DictReader(fh):
            if float(r['value'] or 0) <= 0 or (sel_ids is not None and r['market_id'] not in sel_ids):
                continue
            k, ts = r[key_col], int(r['ts'])
            lo, hi = out.get(k, (ts, ts))
            out[k] = (min(lo, ts), max(hi, ts))
    return out


def window(win, key, start, end):
    """The request range for one id: its active DAY window padded by a day before and two after."""
    w = win.get(key)
    return (max(start, w[0] - 86400), min(end, w[1] + 2 * 86400)) if w else (start, end)


def points(series, warn, label):
    """[(x, y)] from an API timeseries, skipping nulls; warns once if the length looks like a server cap."""
    pts = [(int(float(p['x'])), p['y']) for p in series or [] if p.get('y') is not None]
    if len(series or []) in ROUND_CAPS and label not in warn:
        warn.add(label)
        print(f'  {label}: a request returned exactly {len(series)} points; if the API caps series length, '
              f'lower --chunk-days')
    return pts


def scale(field, y, decimals):
    """API value -> float; token-unit fields are divided by 10**decimals."""
    v = float(y)
    return v / 10 ** int(decimals or 0) if not field.endswith('Usd') else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=str(ROOT / 'config' / 'assets.json'))
    ap.add_argument('--start', default=MORPHO_START)
    ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--interval', default='DAY', choices=['HOUR', 'DAY'])
    ap.add_argument('--chunk-days', type=int, default=0, help='days per history request (default 30 for HOUR, 400 for DAY)')
    ap.add_argument('--hour-min-usd', type=float, default=1e6,
                    help='HOUR pass: only markets that ever held this much (supply or collateral) and vaults that '
                         'ever put a tenth of it into them, judged from the DAY files (0 = everything)')
    ap.add_argument('--sleep', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0, help='only the N largest markets (for a test run)')
    ap.add_argument('--rpc', default=os.environ.get('ETH_RPC_URL', ''),
                    help='Ethereum RPC for reading price-feed descriptions (tells exchange-rate feeds apart); optional')
    ap.add_argument('--out', default=str(ROOT / 'data' / 'morpho'))
    ap.add_argument('--cache', default=str(ROOT / 'data' / 'raw' / 'morpho'))
    ap.add_argument('--reclassify', action='store_true',
                    help='only recompute oracle_class in markets.csv from its feed columns and feed_descriptions.json (no network)')
    args = ap.parse_args()
    if args.reclassify:
        reclassify(args.out)
        return

    _, assets = load_registry(args.registry)
    targets = {a['address'].lower(): a['symbol'] for a in assets if a['in_scope']}
    known = {a['address'].lower(): a['symbol'] for a in assets}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gql = Gql(args.sleep, args.cache)
    start, end = max(to_ts(args.start), to_ts(MORPHO_START)), to_ts(args.end) + 86400
    chunk_days = args.chunk_days or (30 if args.interval == 'HOUR' else 400)
    warn = set()
    hour_filter = args.interval == 'HOUR' and args.hour_min_usd > 0
    if hour_filter and not (out / 'market_history_day.csv.gz').exists():
        print('No DAY files yet: run --interval DAY first, or pass --hour-min-usd 0 to fetch every market hourly.')
        hour_filter = False

    # 1. markets touching registry assets
    items, opt = paginate(gql, q_markets, 'markets', MARKET_OPT)
    sel = [m for m in items if addr(m.get('collateralAsset')) in targets or addr(m.get('loanAsset')) in targets]
    sel.sort(key=lambda m: -float((m.get('state') or {}).get('supplyAssetsUsd') or 0))
    if args.limit:
        sel = sel[:args.limit]
    print(f'{len(items)} Ethereum markets; {len(sel)} touch registry assets')
    feeds = {addr(((m.get('oracle') or {}).get('data') or {}).get(k)) for m in sel for k in ('baseFeedOne', 'baseFeedTwo')}
    desc = read_descriptions(args.rpc, feeds, out / 'feed_descriptions.json')
    rows = []
    for m in sel:
        o = m.get('oracle') or {}
        d = o.get('data') or {}
        col, loan, st = m.get('collateralAsset') or {}, m.get('loanAsset') or {}, m.get('state') or {}
        rows.append({
            'market_id': m['marketId'],
            'collateral': addr(col), 'collateral_symbol': known.get(addr(col), col.get('symbol', '')),
            'collateral_decimals': col.get('decimals', ''),
            'loan': addr(loan), 'loan_symbol': known.get(addr(loan), loan.get('symbol', '')),
            'loan_decimals': loan.get('decimals', ''),
            'lltv': int(m['lltv']) / 1e18 if m.get('lltv') else '',
            'oracle': (o.get('address') or '').lower(), 'oracle_type': o.get('type', '') or d.get('__typename', ''),
            'oracle_class': classify_oracle(o, desc, opt, known.get(addr(col), col.get('symbol', ''))),
            'base_feed_1': addr(d.get('baseFeedOne')), 'base_feed_2': addr(d.get('baseFeedTwo')),
            'base_vault': addr(d.get('baseOracleVault')),
            'quote_feed_1': addr(d.get('quoteFeedOne')), 'quote_feed_2': addr(d.get('quoteFeedTwo')),
            'quote_vault': addr(d.get('quoteOracleVault')),
            'base_feed_desc': ' | '.join(desc.get(addr(d.get(k)), '') for k in ('baseFeedOne', 'baseFeedTwo')
                                         if addr(d.get(k)) not in ('', ZERO)),
            'supply_usd_now': st.get('supplyAssetsUsd', ''), 'borrow_usd_now': st.get('borrowAssetsUsd', ''),
            'collateral_usd_now': st.get('collateralAssetsUsd', '')})
    with open(out / 'markets.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['market_id'])
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r['oracle_class']] = counts.get(r['oracle_class'], 0) + 1
    print(f'oracle classes: {counts}')
    missing = sorted(f for f in feeds if f and f != ZERO and not desc.get(f))
    if missing:
        print(f'  {len(missing)} base feeds have no description' + ('' if args.rpc else
              ' (no ETH_RPC_URL set): exchange-rate feeds count as "feed". Set ETH_RPC_URL and run again to '
              'classify them; the Morpho responses come from the cache'))

    # 2. market history
    hist_sel, mwin = sel, {}
    if hour_filter:
        mx = day_maxima(out / 'market_history_day.csv.gz', 'market_id', {'supplyAssetsUsd', 'collateralAssetsUsd'})
        hist_sel = [m for m in sel if mx.get(m['marketId'], 0) >= args.hour_min_usd]
        mwin = active_windows(out / 'market_history_day.csv.gz', 'market_id')
        print(f'HOUR pass: {len(hist_sel)} of {len(sel)} markets ever held >= ${args.hour_min_usd:,.0f}')
    fields = list(MARKET_FIELDS)
    render = lambda fs: ' '.join(f'{f}(options: $options) {{ x y }}' for f in fs)
    hist = {}
    for i, m in enumerate(hist_sel):
        dec = (m.get('collateralAsset') or {}).get('decimals') or 18
        for a, b in chunks(*window(mwin, m['marketId'], start, end), chunk_days):
            v = {'id': m['marketId'], 'chainId': 1,
                 'options': {'startTimestamp': a, 'endTimestamp': b - 1, 'interval': args.interval}}
            try:
                data, fields = history_query(gql, Q_MARKET_HISTORY, fields, v, render)
            except GqlError as e:
                if VALIDATION.search(str(e)):
                    raise
                print(f"  market {m['marketId'][:12]}: skipped a {chunk_days}-day request ({str(e)[:120]})")
                continue
            hs = ((data or {}).get('marketById') or {}).get('historicalState') or {}
            for f in fields:
                for x, y in points(hs.get(f), warn, 'market history'):
                    hist[(m['marketId'], x, f)] = scale(f, y, dec)
        if (i + 1) % 10 == 0:
            print(f'  market history {i + 1}/{len(hist_sel)} ({gql.n} requests so far)')
    write_gz(out / f'market_history_{args.interval.lower()}.csv.gz', ['market_id', 'ts', 'field', 'value'],
             [(k[0], k[1], k[2], v) for k, v in sorted(hist.items())])
    print(f'market history: {len(hist)} points; fields: {fields}')

    # 3. vaults: lending into the selected markets now, or lending one of their loan assets
    sel_ids = {m['marketId'] for m in sel}
    loan_assets = {addr(m.get('loanAsset')) for m in sel}
    vitems, _ = paginate(gql, q_vaults, 'vaults', VAULT_OPT)
    vsel = []
    for v in vitems:
        st = v.get('state') or {}
        alloc = {(x.get('market') or {}).get('marketId') for x in (st.get('allocation') or [])}
        if alloc & sel_ids:
            vsel.append((v, 'allocation'))
        elif addr(v.get('asset')) in loan_assets:
            vsel.append((v, 'asset'))
    with open(out / 'vaults.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['vault', 'name', 'symbol', 'asset', 'asset_symbol', 'asset_decimals', 'total_assets_usd_now', 'selected_by'])
        for v, why in vsel:
            a = v.get('asset') or {}
            w.writerow([v['address'].lower(), v.get('name', ''), v.get('symbol', ''), addr(a),
                        known.get(addr(a), a.get('symbol', '')), a.get('decimals', ''),
                        (v.get('state') or {}).get('totalAssetsUsd', ''), why])
    print(f'{len(vitems)} Ethereum vaults; {len(vsel)} kept')

    # 4. vault allocation history (allocations to selected markets + vault size)
    vh_sel, vwin = [v for v, _ in vsel], {}
    if hour_filter:
        path = out / 'vault_allocation_day.csv.gz'
        usd = day_maxima(path, 'vault', {'supplyAssetsUsd'}, by_market=True)
        units = day_maxima(path, 'vault', {'supplyAssets'}, by_market=True)     # token units: keep any vault that used them
        keep = {k[0] for k, val in usd.items() if k[1] in sel_ids and val >= args.hour_min_usd / 10}
        keep |= {k[0] for k, val in units.items() if k[1] in sel_ids and val > 0}
        vh_sel = [v for v in vh_sel if v['address'].lower() in keep]
        vwin = active_windows(path, 'vault', sel_ids)
        print(f'HOUR pass: {len(vh_sel)} of {len(vsel)} vaults ever put >= ${args.hour_min_usd / 10:,.0f} into a selected market')
    vfields = list(VAULT_FIELDS)

    def vrender(fs):
        parts = [f'allocation {{ market {{ marketId }} {f}(options: $options) {{ x y }} }}' for f in fs if f.startswith('supply')]
        parts += [f'{f}(options: $options) {{ x y }}' for f in fs if f.startswith('total')]
        return ' '.join(parts)
    vrows = {}
    vchunk = chunk_days
    for i, v in enumerate(vh_sel):
        va, dec = v['address'].lower(), (v.get('asset') or {}).get('decimals') or 18
        for a, b in chunks(*window(vwin, va, start, end), vchunk):
            var = {'address': v['address'], 'chainId': 1,
                   'options': {'startTimestamp': a, 'endTimestamp': b - 1, 'interval': args.interval}}
            try:
                data, vfields = history_query(gql, Q_VAULT_HISTORY, vfields, var, vrender, TWINS)
            except GqlError as e:
                if VALIDATION.search(str(e)):
                    raise
                print(f"  vault {va[:12]}: skipped a {vchunk}-day request ({str(e)[:120]})")
                continue
            hs = ((data or {}).get('vaultByAddress') or {}).get('historicalState') or {}
            for al in hs.get('allocation') or []:
                mid = (al.get('market') or {}).get('marketId')
                if mid not in sel_ids:
                    continue
                for f in vfields:
                    for x, y in points(al.get(f), warn, 'vault history') if f.startswith('supply') else []:
                        vrows[(va, mid, x, f)] = scale(f, y, dec)
            for f in vfields:
                for x, y in points(hs.get(f), warn, 'vault history') if f.startswith('total') else []:
                    vrows[(va, '', x, f)] = scale(f, y, dec)
        if (i + 1) % 10 == 0:
            print(f'  vault history {i + 1}/{len(vh_sel)} ({gql.n} requests so far)')
    write_gz(out / f'vault_allocation_{args.interval.lower()}.csv.gz', ['vault', 'market_id', 'ts', 'field', 'value'],
             [(k[0], k[1], k[2], k[3], v) for k, v in sorted(vrows.items())])
    print(f'vault allocations: {len(vrows)} points; fields: {vfields}; {gql.n} API requests this run')


if __name__ == '__main__':
    main()
