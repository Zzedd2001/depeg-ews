"""Offline tests for the exposure-graph scripts (no network): fetch_morpho, fetch_aave_reserves, build_graph.

python tests/test_exposure.py
"""
import contextlib
import csv
import gzip
import io
import json
import sys
import tempfile
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import build_graph as bg  # noqa: E402
import fetch_aave_reserves as fa  # noqa: E402
import fetch_morpho as fm  # noqa: E402
import fetch_rates_rpc as fr  # noqa: E402
from common import to_ts  # noqa: E402

REG = {a['symbol']: a['address'].lower() for a in json.loads((ROOT / 'config' / 'assets.json').read_text())['assets']}
u256 = fr.u256


def gz_rows(path):
    with gzip.open(path, 'rt') as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- Morpho
F1, F2, Q1 = '0x' + 'f1' * 20, '0x' + 'f2' * 20, '0x' + 'e1' * 20
V1, V2 = 'MorphoChainlinkOracleData', 'MorphoChainlinkOracleV2Data'
REAL_400 = ('HTTP 400: {"errors":[{"message":"Cannot query field \\"description\\" on type \\"OracleFeed\\".",'
            '"status":"GRAPHQL_VALIDATION_FAILED","extensions":{}},{"message":"Cannot query field \\"vendor\\" on type '
            '\\"OracleFeed\\".","status":"GRAPHQL_VALIDATION_FAILED","extensions":{}},{"message":"Cannot query field '
            '\\"description\\" on type \\"OracleFeed\\".","status":"GRAPHQL_VALIDATION_FAILED","extensions":{}}]}')


def test_unknown_field_errors_as_the_api_sends_them():
    # the exact shape of the error body the real API returned (escaped quotes, repeats)
    assert fm.unknown_fields(REAL_400) == [('description', 'OracleFeed'), ('vendor', 'OracleFeed')]
    opt = {'OracleFeed': ['address', 'description', 'vendor'], 'MarketState': ['supplyAssetsUsd']}
    with contextlib.redirect_stdout(io.StringIO()):
        assert fm.drop_unknown(opt, REAL_400) and opt == {'OracleFeed': ['address'], 'MarketState': ['supplyAssetsUsd']}
        assert not fm.drop_unknown(opt, 'Cannot query field "marketId" on type "Market".')   # not optional: re-raise
        assert fm.drop_unknown(opt, 'Unknown type "OracleFeed".') and opt['OracleFeed'] == []
        assert not fm.drop_unknown(opt, 'Internal server error')


def test_classify_oracle():
    Z = fm.ZERO
    c = fm.classify_oracle
    desc = {F1: 'wstETH / stETH Exchange Rate', F2: 'STETH / USD'}
    data = lambda **kw: {'address': '0x1', 'data': {'__typename': V2, **{k: {'address': v} for k, v in kw.items()}}}
    assert c(None) == 'none' and c({'address': Z}) == 'none'
    assert c({'address': '0x1', 'data': None}) == 'custom'
    assert c({'address': '0x1', 'data': {'__typename': 'SomeOtherOracle'}}) == 'custom'
    assert c(data(baseFeedOne=Z, baseOracleVault=Z, quoteFeedOne=Q1)) == 'fixed'
    assert c(data(baseOracleVault='0x' + 'aa' * 20)) == 'vault_rate'
    assert c(data(baseFeedOne=F1), desc) == 'rate_feed'
    assert c(data(baseFeedOne=F1, baseFeedTwo=F2), desc) == 'feed'
    assert c(data(baseFeedOne=F1)) == 'feed'                                   # no description: assume a market price
    # base-side fields missing from the query: the class cannot be judged
    assert c(data(baseFeedOne=Z), available={V2: ['baseFeedOne', 'baseFeedTwo']}) == 'custom'
    # an exchange rate times a reference-asset price is still blind to a depeg of the collateral
    d2 = {F1: 'weETH/ETH exchange rate', F2: 'ETH / USD'}
    assert c(data(baseFeedOne=F1, baseFeedTwo=F2), d2, collateral_symbol='weETH') == 'rate_feed'
    b = fm.base_class
    assert b(['USDC / USD'], 'USDf') == 'fixed' and b(['USDC / USD'], 'USDC') == 'feed'   # USDf valued 1:1 as USDC
    assert b(['ETH / USD'], 'WETH') == 'feed' and b(['WBTC / BTC', 'BTC / USD'], 'WBTC') == 'feed'
    assert b(['ETH / USD'], 'rsETH', vault=True) == 'vault_rate'
    assert b(['RETH / ETH'], 'rETH') == 'feed' and b(['RedStone Price Feed for deUSD_FUNDAMENTAL'], 'deUSD') == 'rate_feed'
    assert b(['wstETH/stETH exchange rate', 'STETH / USD'], 'wstETH') == 'feed'          # sees a stETH depeg
    assert b(['Custom price feed for wstETH / ETH', 'ETH / USD'], 'wstETH') == 'feed'
    assert b(['weETH/ETH exchange rate', ''], 'weETH') == 'feed'                        # one feed unread: no call


def test_reclassify_offline():
    Z = fm.ZERO
    rows = [('m1', 'weETH', '0xf1', '0xf2', '', 'feed'), ('m2', 'wstETH', '0xf3', '0xf4', '', 'feed'),
            ('m3', 'USDf', '0xf5', '', '', 'feed'), ('m4', 'sUSDe', '', '', '0xaa', 'vault_rate'),
            ('m5', 'PT-X', '', '', '', 'custom'), ('m6', 'rsETH', Z, '', Z, 'fixed'), ('m7', 'ezETH', '0xf6', '', '', 'rate_feed')]
    desc = {'0xF1': 'weETH/ETH exchange rate', '0xf2': 'ETH / USD', '0xf3': 'wstETH/stETH exchange rate',
            '0xf4': 'STETH / USD', '0xf5': 'USDC / USD', '0xf6': 'ezETH/ETH exchange rate'}
    with tempfile.TemporaryDirectory() as d:
        with open(Path(d) / 'markets.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['market_id', 'collateral_symbol', 'base_feed_1', 'base_feed_2', 'base_vault', 'oracle_class', 'lltv'])
            w.writerows([r + ('0.915',) for r in rows])
        (Path(d) / 'feed_descriptions.json').write_text(json.dumps(desc))
        with contextlib.redirect_stdout(io.StringIO()):
            ch = fm.reclassify(d)
        got = {r['market_id']: r for r in csv.DictReader(open(Path(d) / 'markets.csv'))}
        assert {k: r['oracle_class'] for k, r in got.items()} == {'m1': 'rate_feed', 'm2': 'feed', 'm3': 'fixed', 'm4': 'vault_rate',
                                                                  'm5': 'custom', 'm6': 'fixed', 'm7': 'rate_feed'}
        assert ch == {'feed -> rate_feed': 1, 'feed -> fixed': 1} and got['m1']['lltv'] == '0.915'


def make_markets():
    usdc, weth = REG['USDC'], REG['WETH']
    asset = lambda a, s, d=18: {'address': a, 'symbol': s, 'decimals': d}
    orc = lambda typ=V2, **kw: {'address': '0x' + 'c0' * 20, 'type': 'ChainlinkOracleV2',
                                'data': {'__typename': typ, **{k: ({'address': v} if v else None) for k, v in kw.items()}}}
    m = [
        ('m1', asset(REG['sUSDe'], 'sUSDe'), asset(usdc, 'USDC', 6), orc(baseOracleVault=REG['sUSDe'], baseFeedOne=None), 5e7),
        ('m2', asset(REG['wstETH'], 'wstETH'), asset(weth, 'WETH'), orc(V1, baseFeedOne=F1), 2e5),
        ('m3', asset(REG['USDe'], 'USDe'), asset(usdc, 'USDC', 6), orc(baseFeedOne=None, quoteFeedOne=Q1), 3e5),
        ('m4', asset('0x' + 'ab' * 20, 'PT-sUSDE-27MAR2025'), asset(REG['USDe'], 'USDe'),
         {'address': '0x' + 'c1' * 20, 'type': 'Custom', 'data': None}, 1e5),
        ('m5', asset('0x' + 'cd' * 20, 'FOO'), asset('0x' + 'ef' * 20, 'BAR'), orc(), 9e9),       # not ours
        ('m6', asset(REG['wstETH'], 'wstETH'), asset(usdc, 'USDC', 6), orc(baseFeedOne=F1, baseFeedTwo=F2), 1e5),
        ('m7', asset(REG['sUSDe'], 'sUSDe'), asset(REG['USDT'], 'USDT', 6), {'address': fm.ZERO, 'type': None, 'data': None}, 1e4),
    ]
    return [{'marketId': mid, 'lltv': str(int(0.915e18)), 'collateralAsset': col, 'loanAsset': loan, 'oracle': o,
             'state': {'supplyAssetsUsd': size, 'borrowAssetsUsd': size / 2, 'collateralAssetsUsd': size * 1.2}}
            for mid, col, loan, o, size in m]


class FakeMorpho:
    """Answers fetch_morpho's four query shapes over HTTP like the real API: a query naming a field the
    schema lacks gets HTTP 400 with one GRAPHQL_VALIDATION_FAILED error per occurrence (JSON-escaped)."""
    MISSING = [('description', 'OracleFeed', 'description'), ('vendor', 'OracleFeed', 'vendor'),
               ('quoteVaultConversionSample', V2, 'quoteVaultConversionSample'),
               ('collateralAssetsUsd', 'MarketHistory', 'collateralAssetsUsd(options'),
               ('supplyAssetsUsd', 'VaultAllocationHistory', 'allocation { market { marketId } supplyAssetsUsd(options')]

    def __init__(self):
        self.markets = make_markets()
        self.posts = []
        usdc = {'address': REG['USDC'], 'symbol': 'USDC', 'decimals': 6}
        self.vaults = [
            {'address': '0xV1', 'name': 'Big USDC', 'symbol': 'bUSDC', 'asset': usdc,
             'state': {'totalAssetsUsd': 1e8, 'allocation': [{'market': {'marketId': 'm1'}, 'supplyAssetsUsd': 4e7}]}},
            {'address': '0xV2', 'name': 'Small USDC', 'symbol': 'sUSDC', 'asset': usdc, 'state': {'totalAssetsUsd': 1e5, 'allocation': []}},
            {'address': '0xV3', 'name': 'DAI vault', 'symbol': 'vDAI', 'asset': {'address': REG['DAI'], 'symbol': 'DAI', 'decimals': 18},
             'state': {'totalAssetsUsd': 1e7, 'allocation': []}},
        ]

    def handle(self, payload):
        q, v = payload['query'], payload['variables']
        self.posts.append(q.split('(')[0])
        errs = [{'message': f'Cannot query field "{f}" on type "{t}".', 'status': 'GRAPHQL_VALIDATION_FAILED', 'extensions': {}}
                for f, t, pat in self.MISSING for _ in range(q.count(pat))]
        if errs:
            return 400, {'errors': errs}
        if q.startswith('query Markets'):
            assert '__typename' in q
            return 200, {'data': {'markets': {'items': self.markets[v['skip']:v['skip'] + v['first']],
                                              'pageInfo': {'countTotal': len(self.markets)}}}}
        if q.startswith('query Vaults'):
            return 200, {'data': {'vaults': {'items': self.vaults[v['skip']:v['skip'] + v['first']],
                                             'pageInfo': {'countTotal': len(self.vaults)}}}}
        o = v['options']
        step = 86400 if o['interval'] == 'DAY' else 3600
        xs = list(range(o['startTimestamp'], o['endTimestamp'] + 1, step))[:5]
        if q.startswith('query H'):
            m = next(m for m in self.markets if m['marketId'] == v['id'])
            dec, size = m['collateralAsset']['decimals'], m['state']['supplyAssetsUsd']
            hs = {'supplyAssetsUsd': [{'x': x, 'y': size} for x in xs], 'borrowAssetsUsd': [{'x': x, 'y': None} for x in xs],
                  'collateralAssets': [{'x': float(x), 'y': str(int(123.5 * 10 ** dec))} for x in xs]}
            res = {'data': {'marketById': {'historicalState': {k: hs[k] for k in hs if k in q}}}}
            if v['id'] == 'm6':      # partial data: the API could not price something; the rest must be kept
                res['errors'] = [{'message': 'No price found for asset 0xf2'}]
            return 200, res
        if q.startswith('query VA'):
            pts = lambda y: [{'x': x, 'y': y} for x in xs]
            alloc = [{'market': {'marketId': 'm5'}, 'supplyAssets': pts('1')}]                   # not a selected market
            if v['address'] == '0xV1':
                alloc.append({'market': {'marketId': 'm1'}, 'supplyAssets': pts(str(4 * 10 ** 13))})   # 40M USDC (6 dp)
            return 200, {'data': {'vaultByAddress': {'historicalState': {'allocation': alloc, 'totalAssetsUsd': pts(1e8)}}}}
        raise AssertionError(q[:60])


class FakeFeedChain(fr.Rpc):
    """Answers description() of the two price feeds through Multicall3, as an RPC endpoint would."""

    def __init__(self):
        super().__init__('fake')
        self.calls = 0

    def batch(self, calls):
        out = []
        for method, params in calls:
            if method == 'eth_blockNumber':
                out.append(hex(20_000_000))
            elif method == 'eth_getBlockByNumber':
                out.append({'timestamp': hex(1_760_000_000)})
            elif method == 'eth_call':
                self.calls += 1
                texts = {F1: 'wstETH / stETH Exchange Rate', F2: 'STETH / USD'}
                sub = decode_agg3_calldata(bytes.fromhex(params[0]['data'][2:]))
                out.append('0x' + encode_agg3_result([(t in texts, enc_string(texts[t]) if t in texts else b'')
                                                      for t, _, d in sub]).hex())
        return out


def run_morpho(api, argv, chain=None):
    old_open, old_page, old_rpc = fm.urllib.request.urlopen, fm.PAGE, fr.Rpc

    def fake_open(req, timeout=0):
        status, body = api.handle(json.loads(req.data.decode()))
        if status != 200:
            raise urllib.error.HTTPError(fm.API, status, 'Bad Request', {}, io.BytesIO(json.dumps(body).encode()))
        return FakeResp(body)
    fm.urllib.request.urlopen, fm.PAGE = fake_open, 3            # force pagination with 7 markets / 3 vaults
    if chain:
        fr.Rpc = lambda url, sleep=0: chain
    sys.argv = ['x'] + argv + (['--rpc', 'fake'] if chain else ['--rpc', ''])
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fm.main()
    finally:
        fm.urllib.request.urlopen, fm.PAGE, fr.Rpc = old_open, old_page, old_rpc
    return buf.getvalue()


def test_morpho_fetch_end_to_end():
    api = FakeMorpho()
    with tempfile.TemporaryDirectory() as d:
        base = ['--start', '2025-01-01', '--end', '2025-01-03', '--sleep', '0', '--out', f'{d}/out', '--cache', f'{d}/cache']
        log = run_morpho(api, base + ['--interval', 'DAY'])
        assert f'API has no field "quoteVaultConversionSample" on {V2}' in log
        assert 'API has no field "collateralAssetsUsd"' in log and 'supplyAssets" (token units)' in log
        assert 'No price found' in log                                    # partial-data warning, data kept
        assert '2 base feeds have no description (no ETH_RPC_URL set)' in log
        mk = list(csv.DictReader(open(f'{d}/out/markets.csv')))
        cls = {r['market_id']: r['oracle_class'] for r in mk}
        assert cls == {'m1': 'vault_rate', 'm2': 'feed', 'm3': 'fixed', 'm4': 'custom', 'm6': 'feed', 'm7': 'none'}
        assert [r['market_id'] for r in mk][:2] == ['m1', 'm3']           # sorted by current supply
        assert {r['collateral_symbol'] for r in mk} >= {'sUSDe', 'PT-sUSDE-27MAR2025'} and float(mk[0]['lltv']) == 0.915
        h = gz_rows(f'{d}/out/market_history_day.csv.gz')
        assert {r['field'] for r in h} == {'supplyAssetsUsd', 'collateralAssets'}      # null borrow points skipped
        col = [r for r in h if r['field'] == 'collateralAssets']
        assert all(abs(float(r['value']) - 123.5) < 1e-9 for r in col)                  # scaled by collateral decimals
        assert {r['market_id'] for r in h} == set(cls)                                   # m6 kept despite the warning
        assert len({r['ts'] for r in h}) == 3                                            # 3 days, no duplicate chunk edges
        vs = {r['vault']: r['selected_by'] for r in csv.DictReader(open(f'{d}/out/vaults.csv'))}
        assert vs == {'0xv1': 'allocation', '0xv2': 'asset'}                             # DAI vault not related
        va = gz_rows(f'{d}/out/vault_allocation_day.csv.gz')
        assert {(r['market_id'], r['field']) for r in va} == {('m1', 'supplyAssets'), ('', 'totalAssetsUsd')}
        assert all(float(r['value']) == 4e7 for r in va if r['field'] == 'supplyAssets')   # 6 decimals applied
        # second run with an RPC: feed descriptions turn m2 (exchange-rate feed only) into rate_feed; Morpho from cache
        n, chain = len(api.posts), FakeFeedChain()
        log = run_morpho(api, base + ['--interval', 'DAY'], chain)
        assert len(api.posts) == n + 3     # only the three rejected first tries (never cached) are sent again
        cls = {r['market_id']: (r['oracle_class'], r['base_feed_desc']) for r in csv.DictReader(open(f'{d}/out/markets.csv'))}
        assert cls['m2'] == ('rate_feed', 'wstETH / stETH Exchange Rate') and cls['m6'][0] == 'feed'
        assert json.loads(open(f'{d}/out/feed_descriptions.json').read())[F2] == 'STETH / USD' and chain.calls == 1
        run_morpho(api, base + ['--interval', 'DAY'], chain)
        assert chain.calls == 1            # descriptions are cached too
        # HOUR pass: only markets that ever held >= $1M in the DAY file (m1: $50M) and vaults with >= $100k in them
        log = run_morpho(api, base + ['--interval', 'HOUR'])
        assert 'HOUR pass: 1 of 6 markets' in log and 'HOUR pass: 1 of 2 vaults' in log
        hh = gz_rows(f'{d}/out/market_history_hour.csv.gz')
        assert {r['market_id'] for r in hh} == {'m1'}


class FakeResp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def test_morpho_rate_limit_handling():
    calls, waits = [], []
    old_open, old_sleep = fm.urllib.request.urlopen, fm.time.sleep

    def fake_open(req, timeout=0):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(fm.API, 429, 'Too Many Requests', {'Retry-After': '7'}, io.BytesIO(b''))
        if len(calls) == 2:
            raise urllib.error.HTTPError(fm.API, 502, 'Bad Gateway', {}, io.BytesIO(b'oops'))
        return FakeResp({'data': {'ok': 1}})
    fm.urllib.request.urlopen, fm.time.sleep = fake_open, waits.append
    try:
        g = fm.Gql(0, tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            assert g.post({'query': 'q'}) == {'data': {'ok': 1}}
        assert waits == [7, 10]                    # Retry-After, then the 5xx back-off of the second attempt
        fm.urllib.request.urlopen = lambda req, timeout=0: (_ for _ in ()).throw(
            urllib.error.HTTPError(fm.API, 429, 'Too Many Requests', {'Retry-After': '604800'}, io.BytesIO(b'')))
        try:
            g.post({'query': 'q'})
            raise AssertionError('should have stopped')
        except SystemExit as e:
            assert '604800' in str(e)
    finally:
        fm.urllib.request.urlopen, fm.time.sleep = old_open, old_sleep


# ---------------------------------------------------------------- Aave / Spark
def test_selectors_match_keccak():
    try:
        from eth_hash.auto import keccak
    except ImportError:
        try:
            from Crypto.Hash import keccak as _k
            keccak = lambda b: _k.new(digest_bits=256, data=b).digest()
        except ImportError:
            print('skip (no keccak library)')
            return
    sigs = {'getReservesList': '()', 'getReserveData': '(address)', 'getConfiguration': '(address)',
            'getEModeCategoryData': '(uint8)', 'getEModeCategoryCollateralConfig': '(uint8)',
            'getEModeCategoryCollateralBitmap': '(uint8)', 'getEModeCategoryBorrowableBitmap': '(uint8)',
            'getEModeCategoryLabel': '(uint8)', 'getReserveTokensAddresses': '(address)', 'getAssetPrice': '(address)',
            'getSourceOfAsset': '(address)', 'BASE_CURRENCY_UNIT': '()', 'totalSupply': '()', 'decimals': '()', 'symbol': '()'}
    assert set(sigs) == set(fa.SEL)
    for name, args in sigs.items():
        assert keccak((name + args).encode())[:4] == fa.SEL[name], name


def config_word(ltv=0, lt=0, bonus=0, dec=18, active=1, frozen=0, borrow=1, paused=0, rf=0, bcap=0, scap=0, emode=0, ceiling=0):
    return (ltv | lt << 16 | bonus << 32 | dec << 48 | active << 56 | frozen << 57 | borrow << 58 | paused << 60 | rf << 64
            | bcap << 80 | scap << 116 | emode << 168 | ceiling << 212)


def test_decode_config():
    c = fa.decode_config(config_word(7200, 7500, 10750, 18, 1, 1, 0, 1, 2000, 1_000_000, 2_000_000_000, 3, 12345678))
    assert (c['ltv'], c['liq_threshold'], c['liq_bonus'], c['decimals']) == (0.72, 0.75, 0.075, 18)
    assert (c['active'], c['frozen'], c['borrowing_enabled'], c['paused'], c['reserve_factor']) == (1, 1, 0, 1, 0.2)
    assert (c['borrow_cap'], c['supply_cap'], c['emode_legacy'], c['debt_ceiling']) == (1_000_000, 2_000_000_000, 3, 123456.78)
    assert fa.decode_config(1 << 255)['ltv'] == 0          # unused high bits do not leak into fields


def enc_string(s, as_bytes32=False):
    b = s.encode()
    if as_bytes32:
        return b.ljust(32, b'\0')
    return u256(0x20) + u256(len(b)) + b + bytes((-len(b)) % 32)


def enc_addr(a):
    return bytes(12) + bytes.fromhex(a[2:])


def decode_agg3_calldata(data):
    assert data[:4] == fr.AGG3
    b = data[4:]
    w = lambda i: int.from_bytes(b[i:i + 32], 'big')
    arr = w(0)
    n, base = w(arr), arr + 32
    out = []
    for i in range(n):
        p = base + w(base + 32 * i)
        q = p + w(p + 64)
        out.append(('0x' + b[p + 12:p + 32].hex(), w(p + 32) != 0, b[q + 32:q + 32 + w(q)]))
    return out


def encode_agg3_result(res):
    tails = [u256(int(ok)) + u256(0x40) + u256(len(d)) + d + bytes((-len(d)) % 32) for ok, d in res]
    offs, cur = [], 32 * len(res)
    for t in tails:
        offs.append(cur)
        cur += len(t)
    return u256(0x20) + u256(len(res)) + b''.join(u256(o) for o in offs) + b''.join(tails)


GENESIS = to_ts('2024-12-31')
A = {k: '0x' + c * 20 for k, c in [('P1', 'f1'), ('P2', 'f2'), ('O1', 'f3'), ('O2', 'f4'), ('D1', 'f5'), ('D2', 'f6'),
                                    ('FEED', 'f8')]}
SUSDE, WSTETH, NEWTOK, USDC = REG['sUSDe'], REG['wstETH'], '0x' + 'a1' * 20, REG['USDC']
LISTED = (to_ts('2025-01-02') - GENESIS) // 12          # NEWTOK is listed on pool 1 from this block


class FakeAaveChain(fr.Rpc):
    """Two pools: P1 behaves like Aave before 3.2 (legacy e-mode), P2 like 3.2+ (bitmaps)."""

    def __init__(self):
        super().__init__('fake')
        self.head_block = (to_ts('2025-01-03T03:00') - GENESIS) // 12
        tokens = lambda i: ('0x' + f'{i:02x}' * 20, '0x' + f'{i + 1:02x}' * 20, '0x' + f'{i + 2:02x}' * 20)
        self.res = {('P1', SUSDE): (0, *tokens(0x10)), ('P1', WSTETH): (1, *tokens(0x20)), ('P1', NEWTOK): (2, *tokens(0x30)),
                    ('P2', USDC): (4, tokens(0x40)[0], '0x' + '0' * 40, tokens(0x40)[2])}
        self.token_owner = {}
        for (p, asset), (rid, a, s, v) in self.res.items():
            self.token_owner[a] = (asset, 'supply')
            self.token_owner[v] = (asset, 'vdebt')
            if int(s, 16):
                self.token_owner[s] = (asset, 'sdebt')

    def ts(self, b):
        return GENESIS + 12 * b

    def batch(self, calls):
        out = []
        for method, params in calls:
            if method == 'eth_blockNumber':
                out.append(hex(self.head_block))
            elif method == 'eth_getBlockByNumber':
                out.append({'timestamp': hex(self.ts(int(params[0], 16)))})
            elif method == 'eth_call':
                block = int(params[1], 16)
                sub = decode_agg3_calldata(bytes.fromhex(params[0]['data'][2:]))
                out.append('0x' + encode_agg3_result([self.call(t, d, block) for t, _, d in sub]).hex())
        return out

    def call(self, target, data, block):
        sel, args = data[:4].hex(), data[4:]
        arg_addr = '0x' + args[12:32].hex() if len(args) >= 32 else None
        arg_int = int.from_bytes(args[:32], 'big') if len(args) >= 32 else None
        pool = {A['P1']: 'P1', A['P2']: 'P2'}.get(target)
        dec = {USDC: 6}.get(arg_addr, 18)
        ok = lambda b: (True, b)
        if pool and sel == 'd1946dbc':
            assets = [a for (p, a) in self.res if p == pool]
            return ok(u256(0x20) + u256(len(assets)) + b''.join(enc_addr(a) for a in assets))
        if pool and sel == '35ea6a75':
            rid, a, s, v = self.res[(pool, arg_addr)]
            return ok(b''.join(u256(x) for x in [0] * 7 + [rid]) + enc_addr(a) + enc_addr(s) + enc_addr(v) + bytes(32 * 4))
        if pool and sel == 'c44b11f7':
            if arg_addr == NEWTOK and block < LISTED:
                return ok(u256(0))
            emode = {SUSDE: 0, WSTETH: 1, NEWTOK: 0, USDC: 0}[arg_addr]
            ltv = {SUSDE: 7200, WSTETH: 7850, NEWTOK: 5000, USDC: 7500}[arg_addr]
            frozen = int(arg_addr == SUSDE and block >= LISTED)
            return ok(u256(config_word(ltv, ltv + 300, 10500, dec, frozen=frozen, scap=2_000_000, emode=emode)))
        if pool == 'P1' and sel == '6c6f6ae1':           # legacy e-mode struct (dynamic: has a label)
            ltv, lt, bonus, label = (9300, 9500, 10100, 'ETH correlated') if arg_int == 1 else (0, 0, 0, '')
            lb = label.encode()
            return ok(u256(0x20) + u256(ltv) + u256(lt) + u256(bonus) + u256(0) + u256(0xa0) + u256(len(lb)) + lb + bytes((-len(lb)) % 32))
        if pool == 'P1':
            return (False, b'')                            # 3.2 getters do not exist before 3.2
        if pool == 'P2' and sel == '6c6f6ae1':
            return ok(u256(0x20) + u256(0) * 4 + u256(0xa0) + u256(0))
        if pool == 'P2' and sel == 'b286f467':
            return ok(u256(9000) + u256(9300) + u256(10200) if arg_int in (1, 2) else bytes(96))
        if pool == 'P2' and sel == 'b0771dba':
            return ok(u256(1 << 4 if arg_int == 2 else 0))
        if pool == 'P2' and sel == '903a2c71':
            return ok(u256(0b11))
        if pool == 'P2' and sel == '2083e183':
            return ok(enc_string({1: 'unused', 2: 'USDC loop'}.get(arg_int, '')))
        if target in (A['O1'], A['O2']):
            if sel == '8c89b64f':
                return ok(u256(10 ** 8))
            if sel == 'b3596f07':
                return ok(u256({SUSDE: 115_000_000, WSTETH: 400_000_000_000, NEWTOK: 50_000_000, USDC: 99_990_000}[arg_addr]))
            if sel == '92bf2be0':
                return ok(enc_addr(A['FEED']))
        if target in self.token_owner:
            asset, kind = self.token_owner[target]
            if asset == NEWTOK and block < LISTED:
                return ok(b'')                             # no code yet: success, empty return data
            d = {USDC: 6}.get(asset, 18)
            amount = {'supply': 1000 + block / 1000, 'vdebt': 400, 'sdebt': 1}[kind]
            return ok(u256(int(amount * 10 ** d)))
        if sel == '313ce567':
            return ok(u256({USDC: 6}.get(target, 18)))
        if sel == '95d89b41':
            return ok(enc_string('NEW', as_bytes32=True) if target == NEWTOK else enc_string('TOKEN'))
        return (False, b'')


def run_aave(chain, pools_path, argv):
    orig = fa.fr.Rpc
    fa.fr.Rpc = lambda url, sleep=0: chain
    sys.argv = ['x', '--rpc', 'https://fake.invalid/v2/key', '--pools', str(pools_path)] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fa.main()
    finally:
        fa.fr.Rpc = orig
    return buf.getvalue()


def test_aave_reader_end_to_end():
    chain = FakeAaveChain()
    with tempfile.TemporaryDirectory() as d:
        pools = Path(d) / 'pools.json'
        pools.write_text(json.dumps({'instances': [
            {'name': 'p1', 'pool': A['P1'], 'data_provider': A['D1'], 'oracle': A['O1'], 'start': '2025-01-01'},
            {'name': 'p2', 'pool': A['P2'], 'data_provider': A['D2'], 'oracle': A['O2'], 'start': '2025-01-02'}]}))
        argv = ['--start', '2025-01-01', '--end', '2025-01-02', '--out', d, '--sleep', '0']
        log = run_aave(chain, pools, argv)
        assert 'p1: 3 reserves, e-mode ids 1..1' in log and 'p2: 1 reserves, e-mode ids 1..2' in log
        rows = list(csv.DictReader(open(f'{d}/aave_reserves.csv')))
        times = sorted({int(r['ts']) for r in rows})
        assert len(times) == 8 and times[0] == to_ts('2025-01-01')
        by = lambda pool, sym: [r for r in rows if r['pool'] == pool and r['symbol'] == sym]
        s = by('p1', 'sUSDe')
        assert len(s) == 8 and float(s[0]['price_usd']) == 1.15 and float(s[0]['ltv']) == 0.72 and float(s[0]['liq_threshold']) == 0.75
        assert float(s[0]['liq_bonus']) == 0.05 and float(s[0]['variable_debt']) == 400 and float(s[0]['stable_debt']) == 1
        assert abs(float(s[0]['supply']) - (1000 + int(s[0]['block']) / 1000)) < 1e-6
        assert [int(r['frozen']) for r in s] == [0, 0, 0, 0, 1, 1, 1, 1]              # frozen from 2025-01-02
        assert s[0]['price_source'] == A['FEED'] and s[0]['supply_cap'] == '2000000'
        assert len(by('p1', 'NEW')) == 4                                               # listed only from 2025-01-02
        u = by('p2', 'USDC')
        assert len(u) == 4 and float(u[0]['supply']) == 1000 + int(u[0]['block']) / 1000 and float(u[0]['stable_debt']) == 0
        em = list(csv.DictReader(open(f'{d}/aave_emode.csv')))
        e1 = [e for e in em if e['pool'] == 'p1']
        assert len(e1) == 8 and all(e['mode'] == 'legacy' and float(e['ltv']) == 0.93 and float(e['liq_bonus']) == 0.01 for e in e1)
        e2 = [e for e in em if e['pool'] == 'p2']
        assert len(e2) == 8 and {e['category'] for e in e2} == {'1', '2'} and all(e['mode'] == 'bitmap' for e in e2)
        assert [e['collateral_bitmap'] for e in e2 if e['category'] == '2'][0] == '0x10'
        meta = json.loads(open(f'{d}/aave_reserves_meta.json').read())
        assert meta['pools']['p2']['emode_labels'] == {'1': 'unused', '2': 'USDC loop'}
        # resume: a second run redoes only the last grid time, without duplicates
        log = run_aave(chain, pools, argv)
        assert 'resuming at 2025-01-02T18:00Z' in log
        rows2 = list(csv.DictReader(open(f'{d}/aave_reserves.csv')))
        assert len(rows2) == len(rows) and len(list(csv.DictReader(open(f'{d}/aave_emode.csv')))) == len(em)
        # --check prints a table and tests archive access
        log = run_aave(chain, pools, ['--check'])
        assert 'sUSDe' in log and 'e-mode   1 ltv 0.930' in log and 'archive access' in log


# ---------------------------------------------------------------- build_graph
PT = '0x' + 'ab' * 20
D = lambda s: to_ts(s)


def write_graph_inputs(d):
    mdir, ldir = Path(d) / 'morpho', Path(d) / 'lending'
    mdir.mkdir()
    ldir.mkdir()
    mk = [('m1', SUSDE, 'sUSDe', USDC, 'USDC', 0.915, 'vault_rate'), ('m2', WSTETH, 'wstETH', REG['WETH'], 'WETH', 0.945, 'rate_feed'),
          ('m3', PT, 'PT-sUSDE-27MAR2025', USDC, 'USDC', 0.86, 'feed'), ('m4', REG['USDe'], 'USDe', USDC, 'USDC', 0.77, 'fixed')]
    pd.DataFrame(mk, columns=['market_id', 'collateral', 'collateral_symbol', 'loan', 'loan_symbol', 'lltv', 'oracle_class']) \
        .to_csv(mdir / 'markets.csv', index=False)
    h = []
    for day in range(1, 11):
        x = D(f'2025-01-{day:02d}')
        h += [('m1', x, 'collateralAssetsUsd', 100e6 if day < 5 else 200e6), ('m1', x, 'borrowAssetsUsd', 80e6),
              ('m1', x, 'supplyAssetsUsd', 90e6), ('m3', x, 'collateralAssetsUsd', 30e6), ('m3', x, 'borrowAssetsUsd', 20e6),
              ('m4', x, 'collateralAssets', 1e6), ('m4', x, 'borrowAssetsUsd', 0.5e6)]
        if day <= 3:
            h += [('m2', x, 'collateralAssetsUsd', 40e6), ('m2', x, 'borrowAssetsUsd', 35e6)]
    pd.DataFrame(h, columns=['market_id', 'ts', 'field', 'value']).to_csv(mdir / 'market_history_day.csv.gz', index=False)
    pd.DataFrame([('0xv1', 'V1', 'v1', USDC, 'USDC', 6, 1e8, 'allocation'), ('0xv2', 'V2', 'v2', REG['WETH'], 'WETH', 18, 1e7, 'allocation')],
                 columns=['vault', 'name', 'symbol', 'asset', 'asset_symbol', 'asset_decimals', 'total_assets_usd_now', 'selected_by']) \
        .to_csv(mdir / 'vaults.csv', index=False)
    va = []
    for day in range(1, 11):
        x = D(f'2025-01-{day:02d}')
        va += [('0xv1', 'm1', x, 'supplyAssetsUsd', 50e6), ('0xv1', 'm3', x, 'supplyAssetsUsd', 20e6), ('0xv1', '', x, 'totalAssetsUsd', 100e6),
               ('0xv2', 'm2', x, 'supplyAssets', 1000.0)]                              # token units (WETH)
    pd.DataFrame(va, columns=['vault', 'market_id', 'ts', 'field', 'value']).to_csv(mdir / 'vault_allocation_day.csv.gz', index=False)
    ar, em = [], []
    for t in range(D('2025-01-01'), D('2025-01-11'), 6 * 3600):
        base = dict(pool='aave_v3_core', time='', block=1, stable_debt=0, price_source='0xfeed', liq_bonus=0.05, reserve_factor=0.1,
                    active=1, borrowing_enabled=1, paused=0, borrow_cap=0, debt_ceiling=0, emode_legacy=0)
        ar.append({**base, 'asset': SUSDE, 'symbol': 'sUSDe', 'ts': t, 'supply': 1e8, 'variable_debt': 0, 'price_usd': 1.15,
                   'ltv': 0.72, 'liq_threshold': 0.75, 'frozen': 0, 'supply_cap': 2e8, 'reserve_id': 5})
        ar.append({**base, 'asset': REG['USDe'], 'symbol': 'USDe', 'ts': t, 'supply': 5e7, 'variable_debt': 1e7, 'price_usd': 1.0,
                   'ltv': 0.0, 'liq_threshold': 0.0, 'frozen': int(t >= D('2025-01-06')), 'supply_cap': 0, 'reserve_id': 6})
        ar.append({**base, 'asset': WSTETH, 'symbol': 'wstETH', 'ts': t, 'supply': 1e4, 'variable_debt': 10, 'price_usd': 4000.0,
                   'ltv': 0.785, 'liq_threshold': 0.81, 'frozen': 0, 'supply_cap': 0, 'reserve_id': 2})
        em.append(dict(pool='aave_v3_core', time='', ts=t, block=1, category=1, ltv=0.93, liq_threshold=0.95, liq_bonus=0.01,
                       collateral_bitmap=hex(1 << 2), borrowable_bitmap='0x0', mode='bitmap'))
        em.append(dict(pool='aave_v3_core', time='', ts=t, block=1, category=2, ltv=0.90, liq_threshold=0.92, liq_bonus=0.03,
                       collateral_bitmap=hex(1 << 5), borrowable_bitmap='0x0', mode='bitmap'))
    pd.DataFrame(ar).reindex(columns=fa.RESERVE_COLS).to_csv(ldir / 'aave_reserves.csv', index=False)
    pd.DataFrame(em).reindex(columns=fa.EMODE_COLS).to_csv(ldir / 'aave_emode.csv', index=False)
    px = []
    for t in range(D('2025-01-01'), D('2025-01-11'), 3600):
        px += [('USDe', t, 0.95), ('sUSDe', t, 1.15), ('wstETH', t, 4000.0), ('WETH', t, 3500.0), ('USDC', t, 1.0)]
    pd.DataFrame(px, columns=['symbol', 'ts', 'price']).to_csv(Path(d) / 'prices.csv.gz', index=False)


def test_build_graph_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        write_graph_inputs(d)
        sys.argv = ['x', '--morpho', f'{d}/morpho', '--lending', f'{d}/lending', '--prices', f'{d}/prices.csv.gz',
                    '--start', '2025-01-01', '--end', '2025-01-10', '--out', f'{d}/graph']
        with contextlib.redirect_stdout(io.StringIO()):
            bg.main()
        f = pd.read_csv(f'{d}/graph/token_features.csv.gz').set_index(['symbol', 'time'])
        g = lambda sym, day, col: f.loc[(sym, f'2025-01-{day:02d}T00:00Z'), col]
        # no look-ahead: the DAY point stamped Jan 5 00:00 (200M) is usable from 01:00, so the first daily snapshot to see it is Jan 6
        assert np.isnan(g('sUSDe', 1, 'mm_collateral_usd'))                  # Jan 1 point only usable from Jan 2
        assert g('sUSDe', 5, 'mm_collateral_usd') == 100e6 and g('sUSDe', 6, 'mm_collateral_usd') == 200e6
        # staleness: wstETH points stop at Jan 3 00:00 (usable from 01:00); still used Jan 6 (71 h old), missing Jan 7
        assert g('wstETH', 6, 'mm_collateral_usd') == 40e6 and np.isnan(g('wstETH', 7, 'mm_collateral_usd'))
        assert g('wstETH', 4, 'mm_pegged_loop_usd') == 35e6 and g('wstETH', 4, 'mm_blind_oracle_share') == 1.0
        assert g('sUSDe', 5, 'mm_blind_oracle_share') == 1.0 and g('sUSDe', 5, 'mm_lltv_wavg') == 0.915
        assert g('sUSDe', 5, 'mm_pegged_loop_usd') == 80e6                    # sUSDe -> USDC: same (USD) peg
        # token units -> USD with the market price: 1M USDe x 0.95
        assert abs(g('USDe', 5, 'mm_collateral_usd') - 0.95e6) < 1e-6
        assert g('sUSDe', 5, 'vault_exposure_usd') == 50e6 and g('sUSDe', 5, 'vault_n') == 1
        assert g('wstETH', 4, 'vault_exposure_usd') == 1000 * 3500.0          # WETH units converted at observation time
        # Aave: e-mode LTV through the bitmap, frozen flag, supply cap, oracle gap vs market price
        assert g('sUSDe', 5, 'aave_ltv_max') == 0.90 and g('wstETH', 5, 'aave_ltv_max') == 0.93
        assert g('sUSDe', 5, 'aave_supply_usd') == 1.15e8 and g('sUSDe', 5, 'aave_supply_cap_use') == 0.5
        assert g('USDe', 5, 'aave_frozen') == 0 and g('USDe', 6, 'aave_frozen') == 1
        assert abs(g('USDe', 5, 'aave_oracle_gap') - (1 / 0.95 - 1)) < 1e-9 and abs(g('sUSDe', 5, 'aave_oracle_gap')) < 1e-12
        assert np.isnan(g('USDe', 5, 'aave_collateral_usd'))                  # liquidation threshold 0: not collateral
        # family of USDe = USDe + sUSDe (wrapper) + PT-sUSDE (derivative of sUSDe)
        fam = 0.95e6 + 100e6 + 30e6 + 1.15e8
        assert abs(g('USDe', 5, 'family_collateral_usd') - fam) < 1e-3
        assert g('USDe', 5, 'family_vault_exposure_usd') == 70e6
        assert abs(g('USDe', 5, 'family_borrow_against_usd') - (0.5e6 + 80e6 + 20e6 + 1e7)) < 1e-3
        st = pd.read_csv(f'{d}/graph/edges_static.csv')
        assert {tuple(r) for r in st[['src', 'dst', 'etype']].to_numpy()} >= {('tok:sUSDe', 'tok:USDe', 'wrapper'),
                                                                             (f'tok:{PT}', 'tok:sUSDe', 'derivative')}
        e = pd.read_csv(f'{d}/graph/edges.csv.gz')
        day5 = e[e['time'] == '2025-01-05T00:00Z']
        assert set(day5['etype']) == {'collateral', 'supply', 'borrow', 'allocation', 'vault_asset', 'pool_supply', 'pool_borrow'}
        c = day5[(day5.src == 'tok:sUSDe') & (day5.etype == 'collateral')].iloc[0]
        assert c['dst'] == 'mm:m1' and c['usd'] == 100e6 and c['oracle_class'] == 'vault_rate'
        n = pd.read_csv(f'{d}/graph/nodes.csv')
        assert {'token', 'morpho_market', 'morpho_vault', 'lending_pool'} <= set(n['ntype'])
        assert n.set_index('node_id').loc[f'tok:{PT}', 'symbol'] == 'PT-sUSDE-27MAR2025'
        # hourly feature table: the same DAY point shows up at 01:00; no edge file
        sys.argv = sys.argv[:-2] + ['--out', f'{d}/graph_h', '--step-hours', '1', '--no-edges']
        with contextlib.redirect_stdout(io.StringIO()):
            bg.main()
        fh = pd.read_csv(f'{d}/graph_h/token_features.csv.gz').set_index(['symbol', 'time'])
        assert fh.loc[('sUSDe', '2025-01-05T00:00Z'), 'mm_collateral_usd'] == 100e6
        assert fh.loc[('sUSDe', '2025-01-05T01:00Z'), 'mm_collateral_usd'] == 200e6
        assert not Path(f'{d}/graph_h/edges.csv.gz').exists() and len(fh) == len(f) * 24


def test_build_graph_cleaning_rules():
    """The two artifacts seen in the first real Morpho run, in miniature: a market whose debt balloons on paper
    after its collateral died (sdeUSD/USDC) and a circular market with unpriced collateral (BONDUSD/USR)."""
    with tempfile.TemporaryDirectory() as d:
        mdir = Path(d) / 'morpho'
        mdir.mkdir()
        BOND = '0x' + 'b0' * 20
        pd.DataFrame([('bad', REG['sdeUSD'], 'sdeUSD', USDC, 'USDC', 0.915, 'feed'), ('circ', BOND, 'BONDUSD', REG['USR'], 'USR', 0.945, 'feed')],
                     columns=['market_id', 'collateral', 'collateral_symbol', 'loan', 'loan_symbol', 'lltv', 'oracle_class']) \
            .to_csv(mdir / 'markets.csv', index=False)
        h, va = [], []
        for day in range(1, 13):
            x = D(f'2025-01-{day:02d}')
            debt = 8e6 * 1.5 ** max(0, day - 5)                 # 8M until day 5, then +50 % a day on paper
            if day not in (8, 9):              # day 8: no USD value at all; day 9: 0 USD with tokens posted (no price)
                h.append(('bad', x, 'collateralAssetsUsd', 10e6 if day < 5 else 0.05e6))
            elif day == 9:
                h.append(('bad', x, 'collateralAssetsUsd', 0.0))
            h += [('bad', x, 'borrowAssetsUsd', debt),
                  ('bad', x, 'supplyAssetsUsd', debt), ('bad', x, 'collateralAssets', 9e6),
                  ('circ', x, 'supplyAssetsUsd', 100e6 + 50e6 * day), ('circ', x, 'borrowAssetsUsd', 100e6 + 50e6 * day),
                  ('circ', x, 'collateralAssets', 1e9)]
            alloc = 5e6 * 1.5 ** max(0, day - 5)
            va += [('0xvb', 'bad', x, 'supplyAssetsUsd', alloc), ('0xvb', '', x, 'totalAssetsUsd', 15e6 + alloc)]
        pd.DataFrame(h, columns=['market_id', 'ts', 'field', 'value']).to_csv(mdir / 'market_history_day.csv.gz', index=False)
        pd.DataFrame([('0xvb', 'VB', 'vb', USDC, 'USDC', 6, 1e7, 'allocation')],
                     columns=['vault', 'name', 'symbol', 'asset', 'asset_symbol', 'asset_decimals', 'total_assets_usd_now', 'selected_by']) \
            .to_csv(mdir / 'vaults.csv', index=False)
        pd.DataFrame(va, columns=['vault', 'market_id', 'ts', 'field', 'value']).to_csv(mdir / 'vault_allocation_day.csv.gz', index=False)
        sys.argv = ['x', '--morpho', str(mdir), '--lending', f'{d}/none', '--prices', f'{d}/none.csv.gz',
                    '--start', '2025-01-01', '--end', '2025-01-12', '--out', f'{d}/graph']
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            bg.main()
        assert 'bad_debt_freeze: 1 markets' in buf.getvalue() and 'unverifiable: 1 markets' in buf.getvalue()
        f = pd.read_csv(f'{d}/graph/token_features.csv.gz').set_index(['symbol', 'time'])
        g = lambda sym, day, col: f.loc[(sym, f'2025-01-{day:02d}T00:00Z'), col]
        # snapshot day d sees the point of day d-1: insolvent from snapshot 6 (point of day 5), frozen from snapshot 8
        assert g('sdeUSD', 6, 'mm_borrow_against_usd') == 8e6 and g('sdeUSD', 7, 'mm_borrow_against_usd') == 12e6
        assert g('sdeUSD', 8, 'mm_borrow_against_usd') == 8e6 and g('sdeUSD', 12, 'mm_borrow_against_usd') == 8e6
        assert g('sdeUSD', 12, 'vault_exposure_usd') == 5e6                         # the vault's share is held too
        assert g('sdeUSD', 10, 'mm_borrow_against_usd') == 8e6                      # missing / zero price: still frozen
        # circular market: counted while young, left out once it has looked circular for 2 days
        assert g('USR', 2, 'mm_supply_usd') == 150e6 and g('USR', 3, 'mm_supply_usd') == 200e6
        assert not g('USR', 4, 'mm_supply_usd') > 0 and not g('USR', 12, 'mm_supply_usd') > 0
        fl = pd.read_csv(f'{d}/graph/market_flags.csv').set_index('rule')
        assert fl.loc['bad_debt_freeze', 'market'] == 'sdeUSD/USDC' and fl.loc['bad_debt_freeze', 'first'] == '2025-01-08T00:00Z'
        assert fl.loc['bad_debt_freeze', 'max_kept_borrow'] == 8e6 and fl.loc['bad_debt_freeze', 'max_raw_borrow'] == 8e6 * 1.5 ** 6
        assert fl.loc['unverifiable', 'first'] == '2025-01-04T00:00Z' and len(fl) == 2
        e = pd.read_csv(f'{d}/graph/edges.csv.gz')
        b = e[(e.time == '2025-01-10T00:00Z') & (e.etype == 'borrow') & (e.src == 'mm:bad')].iloc[0]
        assert b['usd'] == 8e6 and b['flag'] == 'bad_debt_freeze'
        tot = e[(e.time == '2025-01-10T00:00Z') & (e.etype == 'vault_asset')].iloc[0]
        assert tot['usd'] == 20e6                                                    # 15M elsewhere + 5M held
        assert not ((e.time == '2025-01-10T00:00Z') & (e.etype == 'supply') & (e.dst == 'mm:circ')).any()


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print('ok  ', name)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                fails += 1
                print('FAIL', name, repr(e))
    sys.exit(1 if fails else 0)
