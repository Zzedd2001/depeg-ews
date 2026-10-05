"""Offline tests for the fetchers and the DEX-price SQL (no network needed).

python tests/test_fetchers.py   (the SQL test needs: pip install duckdb sqlglot)
"""
import contextlib
import csv
import gzip
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import fetch_llama_prices as fl  # noqa: E402
import fetch_rates_rpc as fr  # noqa: E402


def test_llama_fetch_parses_and_dedupes(tmp=None):
    calls = []

    def fake_http(url, payload=None, **kw):
        calls.append(url)
        if '/prices/first/' in url:
            keys = url.split('/prices/first/')[1].split(',')
            return {'coins': {k: {'symbol': 'X', 'price': 1, 'timestamp': 1735689600} for k in keys}}
        key = url.split('/chart/')[1].split('?')[0]
        start = int(url.split('start=')[1].split('&')[0])
        pts = [{'timestamp': start + 3600 * i + (30 if i % 2 else -45), 'price': 1 - 0.001 * i} for i in range(5)]
        pts.append({'timestamp': start + 3600 * 2 + 1500, 'price': 9.9})   # >20 min from the hour: dropped
        return {'coins': {key: {'symbol': 'X', 'confidence': 0.99, 'decimals': 18, 'prices': pts}}}

    fl.http_json = fake_http
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'p.csv.gz'
        sys.argv = ['x', '--only', 'USDe', '--start', '2025-01-01', '--end', '2025-01-01', '--span', '24',
                    '--sleep', '0', '--cache', str(Path(d) / 'cache'), '--out', str(out)]
        fl.main()
        rows = list(csv.DictReader(gzip.open(out, 'rt')))
        syms = {r['symbol'] for r in rows}
        assert syms == {'USDe', 'WETH'}
        usde = [r for r in rows if r['symbol'] == 'USDe']
        assert [r['hour'] for r in usde] == [f'2025-01-01T0{i}:00Z' for i in range(5)]
        assert all(float(r['price']) < 2 for r in usde)
        n = len(calls)
        fl.main()                                                   # second run is served from the cache
        assert len(calls) == n + 1                                  # only /prices/first is called again


def test_fallback_key_only_fills_missing_hours():
    def fake_http(url, payload=None, **kw):
        if '/prices/first/' in url:
            keys = url.split('/prices/first/')[1].split(',')
            return {'coins': {k: {'symbol': 'X', 'price': 1, 'timestamp': 1735689600} for k in keys}}
        key = url.split('/chart/')[1].split('?')[0]
        start = int(url.split('start=')[1].split('&')[0])
        n, price = (3, 1.0) if key.startswith('ethereum:') else (6, 0.5)   # primary covers 3 h, fallback 6 h
        return {'coins': {key: {'symbol': 'X', 'confidence': 0.99, 'prices':
                                [{'timestamp': start + 3600 * i, 'price': price} for i in range(n)]}}}

    fl.http_json = fake_http
    reg = json.loads((ROOT / 'config' / 'assets.json').read_text())
    with tempfile.TemporaryDirectory() as d:
        regp, out = Path(d) / 'reg.json', Path(d) / 'p.csv.gz'
        reg['assets'] = [a for a in reg['assets'] if a['symbol'] in ('xUSD', 'WETH')]
        regp.write_text(json.dumps(reg))
        sys.argv = ['x', '--registry', str(regp), '--start', '2025-01-01', '--end', '2025-01-01', '--span', '24',
                    '--sleep', '0', '--cache', str(Path(d) / 'cache'), '--out', str(out)]
        fl.main()
        rows = [r for r in csv.DictReader(gzip.open(out, 'rt')) if r['symbol'] == 'xUSD']
        assert [float(r['price']) for r in rows] == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        assert rows[0]['llama_key'].startswith('ethereum:') and rows[-1]['llama_key'] == 'coingecko:staked-stream-usd'


class FakeChain(fr.Rpc):
    """12-second slots with every 7th slot missed."""

    def __init__(self):
        super().__init__('fake')
        self.ts = []
        t, slot = 1_700_000_000, 0
        while len(self.ts) < 5000:
            if slot % 7 != 3:
                self.ts.append(t)
            t += 12
            slot += 1

    def batch(self, calls):
        out = []
        for method, params in calls:
            if method == 'eth_blockNumber':
                out.append(hex(len(self.ts) - 1))
            elif method == 'eth_getBlockByNumber':
                out.append({'timestamp': hex(self.ts[int(params[0], 16)])})
        return out


def test_block_search_handles_missed_slots():
    c = FakeChain()
    for t in [c.ts[100] + 5, c.ts[2500], c.ts[4000] - 1, c.ts[4500] + 11]:
        b = c.block_at(t, hint=len(c.ts) - 1 - (c.ts[-1] - t) // 12)
        assert c.ts[b] <= t < c.ts[b + 1], (t, b)



def encode_results(res):
    """ABI-encode (bool,bytes)[] as Multicall3.aggregate3 returns it."""
    u = lambda x: int(x).to_bytes(32, 'big')
    tails = [u(1 if ok else 0) + u(0x40) + u(len(b)) + b + bytes((-len(b)) % 32) for ok, b in res]
    offs, cur = [], 32 * len(res)
    for t in tails:
        offs.append(cur)
        cur += len(t)
    return u(0x20) + u(len(res)) + b''.join(u(o) for o in offs) + b''.join(tails)


class RateChain(FakeChain):
    """FakeChain that also answers Multicall3 eth_call: every rate is 1 + block / 1e6; fails after `fail_after` calls."""

    def __init__(self, n_rates, fail_after=None):
        super().__init__()
        self.n_rates, self.fail_after, self.calls = n_rates, fail_after, 0

    def batch(self, calls):
        if calls[0][0] != 'eth_call':
            return super().batch(calls)
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError('connection lost')
        b = int(calls[0][1][1], 16)
        val = int((1 + b / 1e6) * 1e18).to_bytes(32, 'big')
        return ['0x' + encode_results([(True, val)] * self.n_rates).hex()]


def test_rates_run_resumes_after_interruption():
    reg = json.loads((ROOT / 'config' / 'assets.json').read_text())
    reg['assets'] = [a for a in reg['assets'] if a['symbol'] in ('wstETH', 'sUSDe')]
    old_rpc = fr.Rpc
    try:
        with tempfile.TemporaryDirectory() as d:
            regp, out = Path(d) / 'reg.json', Path(d) / 'rates.csv.gz'
            regp.write_text(json.dumps(reg))
            t0 = FakeChain().ts[0]
            start, end = pd_hour(t0 + 3600), pd_hour(t0 + 15 * 3600)
            argv = ['x', '--rpc', 'https://fake.invalid/v2/key', '--registry', str(regp), '--start', start, '--end', end, '--step-hours', '1',
                    '--sleep', '0', '--out', str(out)]
            chain = RateChain(2, fail_after=5)
            fr.Rpc = lambda url, sleep: chain
            sys.argv = argv
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    fr.main()
                raise AssertionError('the first run should have failed')
            except RuntimeError:
                pass
            first = list(csv.DictReader(gzip.open(out, 'rt')))
            assert len({r['ts'] for r in first}) == 5
            chain2 = RateChain(2)
            fr.Rpc = lambda url, sleep: chain2
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fr.main()
            assert 'continuing an earlier run' in buf.getvalue()
            rows = list(csv.DictReader(gzip.open(out, 'rt')))
            keys = [(r['symbol'], r['ts']) for r in rows]
            n_grid = len({r['ts'] for r in rows})        # the grid runs to the chain's last block
            assert len(keys) == len(set(keys)) and n_grid > 10 and chain2.calls == n_grid - 4   # 4 times kept, the 5th refetched
            for r in rows:
                b = int(r['block'])
                assert chain2.ts[b] <= int(r['ts']) < chain2.ts[b + 1] and abs(float(r['rate']) - (1 + b / 1e6)) < 1e-12
            # a run killed mid-write leaves a truncated gzip: keep what can be read
            raw = out.read_bytes()
            out.write_bytes(raw[:len(raw) // 2])
            kept, resume = fr.previous_rows(out)
            assert 0 < len(kept) < len(rows) and resume is not None and all(int(r[2]) < resume for r in kept)
    finally:
        fr.Rpc = old_rpc



class FlakyChain(RateChain):
    """Answers like RateChain, but the proxy drops eth_calls number `drops` (a set of call counts)."""

    def __init__(self, n_rates, drops):
        super().__init__(n_rates)
        self.drops, self.tries = set(drops), 0

    def batch(self, calls):
        if calls[0][0] == 'eth_call':
            self.tries += 1
            if self.tries in self.drops:
                import urllib.error
                raise urllib.error.URLError('Tunnel connection failed: 503 Service Unavailable')
        return super().batch(calls)


def test_rates_run_waits_out_network_drops():
    reg = json.loads((ROOT / 'config' / 'assets.json').read_text())
    reg['assets'] = [a for a in reg['assets'] if a['symbol'] in ('wstETH', 'sUSDe')]
    old_rpc, old_waits = fr.Rpc, fr.RETRY_WAITS
    fr.RETRY_WAITS = (0, 0, 0)
    try:
        with tempfile.TemporaryDirectory() as d:
            regp, out = Path(d) / 'reg.json', Path(d) / 'rates.csv.gz'
            regp.write_text(json.dumps(reg))
            t0 = FakeChain().ts[0]
            argv = ['x', '--rpc', 'https://fake.invalid/v2/key', '--registry', str(regp), '--start', pd_hour(t0 + 3600),
                    '--end', pd_hour(t0 + 15 * 3600), '--step-hours', '1', '--sleep', '0', '--out', str(out)]
            chain = FlakyChain(2, drops={3, 4, 9})          # two drops in a row, then one more later
            fr.Rpc = lambda url, sleep: chain
            sys.argv = argv
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fr.main()
            assert buf.getvalue().count('network error') == 3 and 'fake.invalid' not in buf.getvalue()
            rows = list(csv.DictReader(gzip.open(out, 'rt')))
            assert len({r['ts'] for r in rows}) > 10 and all(r['ok'] == '1' for r in rows)
            # a connection that never comes back: a clear exit, the rows so far kept, no URL in the message
            dead = FlakyChain(2, drops=set(range(6, 100)))
            fr.Rpc = lambda url, sleep: dead
            sys.argv = argv + ['--restart']
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    fr.main()
                raise AssertionError('should have stopped')
            except SystemExit as e:
                msg = str(e)
            assert 'continues from' in msg and 'fake.invalid' not in msg
            kept, resume = fr.previous_rows(out)
            assert len({r[2] for r in kept}) == 4 and resume is not None
    finally:
        fr.Rpc, fr.RETRY_WAITS = old_rpc, old_waits


def pd_hour(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts - ts % 3600, datetime.timezone.utc).strftime('%Y-%m-%dT%H:00')


def test_multicall_roundtrip_against_eth_abi():
    try:
        from eth_abi import encode
    except ImportError:
        print('skip (eth_abi not installed)')
        return
    calls = [('0x72D07D7DcA67b8A406aD1Ec34ce969c90bFEE768', True, bytes.fromhex('679aefce')),
             ('0x0655977FEb2f289A4aB78af67BAB0d17aAb84367', True, fr.rate_calldata({'selector': '0x07a2d13a', 'arg': '1e18'}))]
    assert fr.encode_aggregate3(calls) == bytes.fromhex('82ad56cb') + encode(['(address,bool,bytes)[]'], [calls])
    res = [(True, (12 * 10 ** 17).to_bytes(32, 'big')), (False, b'')]
    assert fr.decode_aggregate3(encode(['(bool,bytes)[]'], [res])) == res


def test_dex_sql_logic_with_duckdb():
    try:
        import duckdb
        import sqlglot
    except ImportError:
        print('skip (duckdb/sqlglot not installed)')
        return
    reg = json.loads((ROOT / 'config' / 'assets.json').read_text())
    dec_path = ROOT / 'config' / 'decimals.json'
    had = dec_path.exists()
    backup = dec_path.read_text() if had else None
    dec_path.write_text(json.dumps({a['symbol']: (6 if a['symbol'] in ('USDC', 'USDT') else 18) for a in reg['assets']}))
    public = '`bigquery-public-data.goog_blockchain_ethereum_mainnet_us.token_transfers`'
    try:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'make_sql.py')], check=True, capture_output=True)
        y23 = (ROOT / 'sql' / 'generated' / '02_dex_trades_2023.sql').read_text()
        y26 = (ROOT / 'sql' / 'generated' / '02_dex_trades_2026.sql').read_text()
        assert 'CREATE OR REPLACE TABLE `depeg.dex_trades_2023`' in y23 and y23.count(public) == 1   # one scan of the public table
        assert "TIMESTAMP('2026-01-01')" in y26 and "TIMESTAMP('2026-10-01')" in y26
        assert 'DELETE' not in y23 + y26 and 'PARTITION BY' not in y23 + y26   # no DML; no partitions (the sandbox expires old ones)
        assert all(f"'{a['address'].lower()}'" in y23 for a in reg['assets'] if a['in_scope'])
        assert not re.search(r'\{[a-z_]+\}', y23 + y26)   # every placeholder rendered
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'make_sql.py'), '--project', 'p', '--dataset', 'd'],
                       check=True, capture_output=True)
        y25 = (ROOT / 'sql' / 'generated' / '02_dex_trades_2025.sql').read_text()
        prices = (ROOT / 'sql' / 'generated' / '03_dex_prices_hourly.sql').read_text()
        assert prices.count('`p.d.dex_trades_*`') == 1 and 'token_transfers`' not in prices
        y25 = y25.replace(public, 'token_transfers').replace('`p.d.dex_trades_2025`', 'dex_trades_2025')
        prices = prices.replace('`p.d.dex_trades_*`', 'dex_trades_all')
    finally:
        dec_path.write_text(backup) if had else dec_path.unlink()
        import shutil
        shutil.rmtree(ROOT / 'sql' / 'generated', ignore_errors=True)
    A = {a['symbol']: a['address'].lower() for a in reg['assets']}
    Z = '0x' + '0' * 40
    U, R, P, P2, L = ['0x' + c * 40 for c in 'abcde']
    tx = {'t0': (99, 0), 't1': (100, 0), 't2': (100, 1), 't3': (101, 0), 't4': (101, 1), 't5': (102, 0), 't6': (103, 0), 't7': (103, 1),
          't8': (104, 0)}
    rows = [
        ('t0', '2024-12-31 23:30:00', U, P, A['USDC'], 1000e6), ('t0', '2024-12-31 23:30:00', P, U, A['USDe'], 900e18),  # before 2025
        ('t1', '2025-01-01 10:05:00', U, P, A['USDC'], 1000e6), ('t1', '2025-01-01 10:05:00', P, U, A['USDe'], 1001e18),
        ('t2', '2025-01-01 10:20:00', U, R, A['USDC'], 500e6), ('t2', '2025-01-01 10:20:00', R, P, A['USDC'], 500e6),
        ('t2', '2025-01-01 10:20:00', P, R, A['USDe'], 500.5e18), ('t2', '2025-01-01 10:20:00', R, U, A['USDe'], 500.5e18),
        ('t3', '2025-01-01 10:30:00', U, A['sUSDe'], A['USDe'], 100e18), ('t3', '2025-01-01 10:30:00', Z, U, A['sUSDe'], 90e18),
        ('t4', '2025-01-01 10:40:00', U, P2, A['WETH'], 10e18), ('t4', '2025-01-01 10:40:00', P2, U, A['wstETH'], 9.9e18),
        ('t5', '2025-01-01 10:50:00', U, P, A['USDC'], 50e6), ('t5', '2025-01-01 10:50:00', P, U, A['USDe'], 50e18),
        # split route at 11:10: 600 USDC to pool P, 400 USDC to pool P2; the trader receives 1001.5 USDe in total
        ('t6', '2025-01-01 11:10:00', U, R, A['USDC'], 1000e6), ('t6', '2025-01-01 11:10:00', R, P, A['USDC'], 600e6),
        ('t6', '2025-01-01 11:10:00', R, P2, A['USDC'], 400e6), ('t6', '2025-01-01 11:10:00', P, R, A['USDe'], 601e18),
        ('t6', '2025-01-01 11:10:00', P2, R, A['USDe'], 400.5e18), ('t6', '2025-01-01 11:10:00', R, U, A['USDe'], 1001.5e18),
        ('t7', '2025-01-01 11:20:00', U, P, A['USDC'], None), ('t7', '2025-01-01 11:20:00', P, U, A['USDe'], 70e18),   # unparsed amount
        # a loan, not a trade: 1000 USDe deposited with lender L, 900 USDC borrowed (pairs at the loan-to-value ratio)
        ('t8', '2025-01-01 10:45:00', U, L, A['USDe'], 1000e18), ('t8', '2025-01-01 10:45:00', L, U, A['USDC'], 900e6),
    ]
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute('CREATE TABLE token_transfers (block_number BIGINT, transaction_index BIGINT, block_timestamp TIMESTAMPTZ, '
                'address VARCHAR, from_address VARCHAR, to_address VARCHAR, quantity VARCHAR)')
    con.executemany('INSERT INTO token_transfers VALUES (?, ?, ?, ?, ?, ?, ?)',
                    [(*tx[t], ts + '+00', tok, a, b, 'n/a' if q is None else f'{q:.0f}') for t, ts, a, b, tok, q in rows])
    for stmt in sqlglot.transpile(y25, read='bigquery', write='duckdb'):
        con.execute(stmt)
    trades = con.execute('SELECT * FROM dex_trades_2025 ORDER BY block_number, transaction_index').fetchdf()
    assert [tuple(r) for r in trades[['block_number', 'transaction_index', 'symbol', 'quote']].to_numpy()] == [
        (100, 0, 'USDe', 'USDC'), (100, 1, 'USDe', 'USDC'), (101, 1, 'wstETH', 'WETH'), (102, 0, 'USDe', 'USDC'), (103, 0, 'USDe', 'USDC'),
        (104, 0, 'USDe', 'USDC')]
    assert trades.set_index('block_number').loc[103, 'asset_amount'] == 1001.5   # split route: the trader's total
    con.execute('CREATE VIEW dex_trades_all AS SELECT * FROM dex_trades_2025')
    out = con.execute(sqlglot.transpile(prices, read='bigquery', write='duckdb')[0]).fetchdf()
    df = out.set_index(['symbol', 'quote', 'hour'])
    assert set(df.index.get_level_values(0)) == {'USDe', 'wstETH'}  # sUSDe mint and dust trade ignored
    h10, h11 = ('USDe', 'USDC', '2025-01-01T10:00Z'), ('USDe', 'USDC', '2025-01-01T11:00Z')
    assert df.loc[h10, 'n_trades'] == 3                                   # t1, t2 and the loan; the $50 trade is below the minimum
    assert abs(df.loc[h10, 'vwap'] - 2400 / 2501.5) < 1e-9                # the loan drags the volume-weighted price to 0.96
    assert abs(df.loc[h10, 'p50'] - 1000 / 1001) < 1e-9 and abs(df.loc[h10, 'p75'] - 1000 / 1001) < 1e-9   # the quartiles do not move
    assert abs(df.loc[h11, 'vwap'] - 1000 / 1001.5) < 1e-9 and df.loc[h11, 'n_trades'] == 1   # split route = one trade
    w = df.loc[('wstETH', 'WETH', '2025-01-01T10:00Z')]
    assert abs(w['vwap'] - 10 / 9.9) < 1e-9 and abs(w['p50'] - 10 / 9.9) < 1e-9 and w['unit'] == 'ETH'
    sys.path.insert(0, str(ROOT / 'scripts'))
    import make_labels as ml
    hourly = ml.collapse_dex_quotes(out.assign(source='dex')).set_index(['symbol', 'hour'])
    assert abs(hourly.loc[('USDe', '2025-01-01T10:00Z'), 'price'] - 1000 / 1001) < 1e-9


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print('ok  ', name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print('FAIL', name, repr(e))
    sys.exit(1 if fails else 0)
