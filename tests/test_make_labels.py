"""Tests for the label rules: python -m pytest -q tests (or python tests/test_make_labels.py)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import make_labels as ml  # noqa: E402

RULES = json.loads((ROOT / 'config' / 'label_rules.json').read_text())
IDX = pd.date_range('2025-01-01', periods=400, freq='h', tz='UTC')


def det(dev, theta=0.01, peg_rules=RULES):
    return ml.detect(dev, theta, peg_rules['min_consecutive_hours'], peg_rules['single_hour_severe'],
                     peg_rules['recovery_band_frac'], peg_rules['recovery_hours'])


def flat(values=None):
    s = pd.Series(0.0, index=IDX)
    for i, v in (values or {}).items():
        s.iloc[i] = v
    return s


def test_minor_dip_is_one_episode():
    eps = det(flat({100: -0.02, 101: -0.02, 102: -0.02}))
    assert len(eps) == 1
    e = eps[0]
    assert e['start'] == IDX[100] and e['end'] == IDX[102] and e['hours_below'] == 3
    assert abs(e['min_dev'] + 0.02) < 1e-12 and not e['ongoing']


def test_single_moderate_hour_is_ignored_but_severe_hour_counts():
    assert det(flat({100: -0.015})) == []
    eps = det(flat({100: -0.13}))
    assert len(eps) == 1 and eps[0]['start'] == IDX[100]


def test_dips_closer_than_recovery_window_merge():
    eps = det(flat({100: -0.02, 101: -0.02, 110: -0.03, 111: -0.03}))
    assert len(eps) == 1 and eps[0]['end'] == IDX[111]


def test_dips_after_full_recovery_are_separate():
    assert len(det(flat({100: -0.02, 101: -0.02, 140: -0.03, 141: -0.03}))) == 2


def test_missing_hour_does_not_break_a_run():
    s = flat({100: -0.02, 102: -0.02})
    s.iloc[101] = np.nan
    assert len(det(s)) == 1


def test_episode_open_at_series_end_is_ongoing():
    eps = det(flat({i: -0.3 for i in range(390, 400)}))
    assert len(eps) == 1 and eps[0]['ongoing']


def test_labels_mark_the_hours_before_a_start():
    dev = flat({200: -0.02, 201: -0.02})
    lab = ml.labels_for(dev, det(dev), [24, 72])
    assert lab['y24'].iloc[176:200].eq(1).all()
    assert lab['y24'].iloc[175] == 0
    assert lab['y72'].iloc[128] == 1 and lab['y72'].iloc[127] == 0
    assert lab['y24'].iloc[200:202].isna().all() and lab['in_episode'].iloc[200] == 1
    assert lab['y24'].iloc[-1:].isna().all()  # right-censored


def test_rate_reference_uses_only_past_rates():
    asset = {'symbol': 'LST', 'peg': 'ETH', 'reference': {'kind': 'rate'}}
    hours = IDX[:300]
    rate = pd.Series(1.10, index=hours)
    rate.iloc[150:] = 1.20                       # rate jumps at hour 150
    ratio = rate.copy()
    ratio.iloc[100:105] = 1.10 * 0.97            # 3% discount for 5 hours
    prices = pd.concat([
        pd.DataFrame({'symbol': 'LST', 'hour': hours, 'price': ratio.values * 3000, 'source': 'x'}),
        pd.DataFrame({'symbol': 'WETH', 'hour': hours, 'price': 3000.0, 'source': 'x'})])
    rates = pd.DataFrame({'symbol': 'LST', 'time': hours[::6], 'rate': rate.values[::6]})
    dev, kind = ml.deviation(asset, prices, rates, RULES, 'x')
    assert kind == 'rate'
    assert abs(dev.iloc[102] + 0.03) < 1e-9      # discount measured against the rate known at the time
    assert abs(dev.iloc[149]) < 1e-9             # before the jump: old rate, no look-ahead
    eps = det(dev, theta=RULES['threshold']['ETH'])
    assert len(eps) == 1 and eps[0]['start'] == hours[100]


def test_trailing_median_catches_a_sudden_drop_not_slow_growth():
    asset = {'symbol': 'Y', 'peg': 'USD', 'reference': {'kind': 'trailing_median'}}
    hours = IDX
    price = pd.Series(1.0 * (1 + 0.10) ** (np.arange(len(hours)) / 8760), index=hours)  # 10% a year
    price.iloc[300:305] *= 0.92
    prices = pd.DataFrame({'symbol': 'Y', 'hour': hours, 'price': price.values, 'source': 'x'})
    dev, _ = ml.deviation(asset, prices, None, RULES, 'x')
    eps = det(dev)
    assert len(eps) == 1 and eps[0]['start'] == hours[300]


def test_eth_ratio_interpolates_weth_at_the_asset_time():
    # WETH observed on the hour and rising 3% an hour; the LST observed 25 minutes later at exactly
    # the interpolated WETH price. Hour-aligned division would show a fake ~1.2% discount.
    hours = IDX[:48]
    t0 = hours.asi8 // 10**9
    weth = 3000 * 1.03 ** np.arange(48)
    lst_ts = t0 + 1500
    lst = np.interp(lst_ts, t0, weth)
    prices = pd.concat([
        pd.DataFrame({'symbol': 'WETH', 'hour': hours, 'ts': t0, 'price': weth, 'source': 'x', 'unit': 'USD'}),
        pd.DataFrame({'symbol': 'LST', 'hour': hours, 'ts': lst_ts, 'price': lst, 'source': 'x', 'unit': 'USD'})])
    r = ml.eth_ratio(prices, 'LST', 'x', 3)
    assert np.allclose(r.iloc[:-1].to_numpy(), 1.0)          # last point has no WETH after it: carried over
    naive = (prices[prices.symbol == 'LST'].price.to_numpy() / weth)
    assert naive[:-1].max() > 1.01                              # what hour-aligned division would report


def fixture(symbol):
    df = pd.read_csv(ROOT / 'tests' / 'fixtures' / 'defillama_snippets.csv')
    df = df[df['symbol'] == symbol].copy()
    df['hour'] = pd.to_datetime(df['hour'], utc=True)
    s = df.set_index('hour')['price'].sort_index()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h')).ffill(limit=RULES['max_gap_fill_hours'])


def test_real_ezeth_crash_hour_is_detected():
    ratio = fixture('ezETH') / fixture('WETH')
    ref = ratio.iloc[:7].median()                 # pre-crash ezETH/ETH level as the reference
    eps = det(ratio / ref - 1, theta=RULES['threshold']['ETH'])
    assert len(eps) == 1
    assert eps[0]['start'] == pd.Timestamp('2024-04-24T03:00Z') and eps[0]['min_dev'] < -0.13


def test_real_usde_cex_print_is_not_an_onchain_episode():
    assert det(fixture('USDe') - 1) == []         # single hourly sample at -2.8%


def test_real_busd0_floor_change_is_detected():
    eps = det(fixture('bUSD0') - 1)
    assert len(eps) == 1
    assert eps[0]['start'] == pd.Timestamp('2025-01-10T00:00Z') and eps[0]['min_dev'] < -0.09


def test_real_deusd_collapse_is_detected_from_daily_points():
    eps = det(fixture('deUSD') - 1)
    assert len(eps) == 1 and eps[0]['start'] == pd.Timestamp('2025-11-07T00:00Z')
    assert ml.severity(eps[0]['min_dev'], RULES['severity_tiers']) == 'collapse'


def test_end_to_end_run_writes_all_outputs():
    reg = {'chain': 'ethereum', 'multicall3': '0x0', 'assets': [
        {'symbol': 'AAA', 'category': 'synthetic', 'peg': 'USD', 'reference': {'kind': 'fixed', 'value': 1.0}, 'in_scope': True},
        {'symbol': 'LST', 'category': 'lst', 'peg': 'ETH', 'reference': {'kind': 'eth_parity'}, 'in_scope': True},
        {'symbol': 'WETH', 'category': 'control', 'peg': 'ETH', 'reference': {'kind': 'fixed', 'value': 1.0}, 'in_scope': False}]}
    hours = IDX
    rng = np.random.default_rng(0)
    aaa = 1 + rng.normal(0, 0.0005, len(hours))
    aaa[50:53] = 0.97
    aaa[250:280] = 0.5
    weth = 3000 * (1 + rng.normal(0, 0.01, len(hours)))
    lst = weth * (1 + rng.normal(0, 0.0005, len(hours)))
    lst[120:124] = weth[120:124] * 0.96
    df = pd.concat([pd.DataFrame({'symbol': 'AAA', 'hour': hours.strftime('%Y-%m-%dT%H:00Z'), 'price': aaa}),
                    pd.DataFrame({'symbol': 'LST', 'hour': hours.strftime('%Y-%m-%dT%H:00Z'), 'price': lst}),
                    pd.DataFrame({'symbol': 'WETH', 'hour': hours.strftime('%Y-%m-%dT%H:00Z'), 'price': weth})])
    df['confidence'] = 0.99
    df['source'] = 'defillama'
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'reg.json').write_text(json.dumps(reg))
        (d / 'known.json').write_text(json.dumps([{'symbol': 'AAA', 'date': '2025-01-11', 'expect': 'onchain', 'note': 't'}]))
        df.to_csv(d / 'p.csv', index=False)
        args = type('A', (), {'prices': [str(d / 'p.csv')], 'primary': '', 'rates': '', 'registry': str(d / 'reg.json'),
                              'rules': str(ROOT / 'config' / 'label_rules.json'), 'known': str(d / 'known.json'),
                              'out_dir': str(d / 'out')})
        ml.run(args)
        ep = pd.read_csv(d / 'out' / 'episodes.csv')
        assert sorted(ep['symbol']) == ['AAA', 'AAA', 'LST']
        assert set(ep['severity']) == {'minor', 'collapse'}
        lab = pd.read_csv(d / 'out' / 'labels_hourly.csv.gz')
        assert {'symbol', 'hour', 'dev', 'in_episode', 'y24', 'y72'} <= set(lab.columns)
        kn = pd.read_csv(d / 'out' / 'known_events_check.csv')
        assert kn.loc[0, 'detected'] == 'yes'
        for f in ['coverage.csv', 'sensitivity.csv', 'checkpoint_report.md']:
            assert (d / 'out' / f).exists()
        assert '## Episodes: 3' in (d / 'out' / 'checkpoint_report.md').read_text()


def run_synthetic(series, tmp, known=None, fill_counts=False):
    """series: {symbol: (peg, reference_kind, prices array)} on IDX (NaN = no price that hour); runs make_labels
    end to end."""
    reg = {'chain': 'ethereum', 'multicall3': '0x0', 'assets': [
        {'symbol': sym, 'category': 'lst' if peg == 'ETH' else 'synthetic', 'peg': peg,
         'reference': {'kind': kind, 'value': 1.0}, 'in_scope': sym != 'WETH'} for sym, (peg, kind, _) in series.items()]}
    frames = [pd.DataFrame({'symbol': sym, 'hour': IDX.strftime('%Y-%m-%dT%H:00Z'), 'price': arr}).dropna()
              for sym, (_, _, arr) in series.items()]
    df = pd.concat(frames)
    df['confidence'] = 0.99
    df['source'] = 'defillama'
    tmp = Path(tmp)
    (tmp / 'reg.json').write_text(json.dumps(reg))
    (tmp / 'known.json').write_text(json.dumps(known or []))
    df.to_csv(tmp / 'p.csv', index=False)
    args = type('A', (), {'prices': [str(tmp / 'p.csv')], 'primary': '', 'rates': '', 'registry': str(tmp / 'reg.json'),
                          'rules': str(ROOT / 'config' / 'label_rules.json'), 'known': str(tmp / 'known.json'),
                          'out_dir': str(tmp / 'out'), 'fill_counts': fill_counts})
    ml.run(args)
    ep = pd.read_csv(tmp / 'out' / 'episodes.csv', parse_dates=['start', 'end']).fillna({'suspect': ''})
    lab = pd.read_csv(tmp / 'out' / 'labels_hourly.csv.gz', parse_dates=['hour'])
    return ep, lab


def noisy(n, seed):
    return 1 + np.random.default_rng(seed).normal(0, 0.0005, n)


def test_snapback_print_is_flagged_and_gives_no_labels():
    a = noisy(len(IDX), 1)
    a[100:104] = 0.111                      # like mkUSD on 2023-10-06: 0.11 for four hours, then back
    b = noisy(len(IDX), 2)
    b[200:203] = 0.97                       # an ordinary dip
    with tempfile.TemporaryDirectory() as d:
        ep, lab = run_synthetic({'AAA': ('USD', 'fixed', a), 'BBB': ('USD', 'fixed', b)}, d)
    sus = ep[ep.symbol == 'AAA']
    assert len(sus) == 1 and sus.iloc[0]['suspect'] == 'snapback'
    la = lab[lab.symbol == 'AAA'].set_index('hour')
    assert la['y72'].iloc[30:104].isna().all()                  # run-up and the print itself carry no label
    assert ep[ep.symbol == 'BBB'].iloc[0]['suspect'] == ''


def test_one_hour_drops_in_three_assets_at_once_are_flagged():
    series = {}
    for i, sym in enumerate(['AAA', 'BBB', 'CCC']):
        x = noisy(len(IDX), 10 + i)
        x[150] = 0.88                            # same hour, one sample each (like 2024-10-04 08:00)
        series[sym] = ('USD', 'fixed', x)
    with tempfile.TemporaryDirectory() as d:
        ep, _ = run_synthetic(series, d)
    assert len(ep) == 3 and set(ep['suspect']) == {'same_hour_cluster'}


def test_collapsed_asset_stops_being_a_target():
    x = noisy(len(IDX), 3)
    x[100:180] = 0.10                            # collapse, 80 hours below
    x[180:] = 0.10 * (1 + 0.5 * np.sin(np.arange(len(IDX) - 180) / 5))  # a dead token keeps swinging
    with tempfile.TemporaryDirectory() as d:
        ep, lab = run_synthetic({'XXX': ('USD', 'trailing_median', x)}, d)
    assert len(ep) == 1 and bool(ep.iloc[0]['terminal'])          # later swings are not new episodes
    lx = lab.set_index('hour')
    assert lx['y24'].iloc[100:].isna().all() and lx['y24'].iloc[80:100].eq(1).all()


def test_one_bad_print_inside_a_mild_episode_is_not_a_collapse():
    x = noisy(len(IDX), 4)
    x[100:160] = 0.985                           # 60 hours at -1.5% (like DOLA in Dec 2023)
    x[130] = 0.10                                # one bad print inside
    with tempfile.TemporaryDirectory() as d:
        ep, lab = run_synthetic({'DDD': ('USD', 'fixed', x)}, d)
    assert len(ep) == 1
    e = ep.iloc[0]
    assert e['severity'] == 'minor' and not bool(e['terminal']) and e['min_dev'] < -0.8
    assert lab.set_index('hour')['y24'].iloc[200:].notna().any()   # still a target afterwards


def test_a_single_last_sample_does_not_set_the_severity():
    x = noisy(len(IDX), 5)
    x[100:105] = 0.98                            # five hours at -2%
    x[105] = 0.92                                # one sample at -8%, then back to the peg
    with tempfile.TemporaryDirectory() as d:
        ep, _ = run_synthetic({'EEE': ('USD', 'fixed', x)}, d)
    assert len(ep) == 1
    e = ep.iloc[0]
    assert e['min_dev'] < -0.07 and abs(e['held_min_dev'] + 0.02) < 0.002 and e['severity'] == 'minor'


def test_a_one_sample_spike_opens_an_episode_but_is_minor():
    x = noisy(len(IDX), 6)
    x[100] = 0.80                                # one sample at -20 %, then back to the peg (like tETH, 2026-08-30)
    with tempfile.TemporaryDirectory() as d:
        ep, lab = run_synthetic({'FFF': ('USD', 'fixed', x)}, d)
    assert len(ep) == 1
    e = ep.iloc[0]
    assert e['min_dev'] < -0.19 and np.isnan(e['held_min_dev']) and e['severity'] == 'minor'
    assert lab.set_index('hour')['y24'].iloc[80:100].eq(1).all()   # it still counts as an onset


def test_a_single_print_before_a_gap_does_not_start_an_episode():
    x = noisy(len(IDX), 21)
    x[100] = 0.985                               # one print at -1.5 %, then three hours without a price
    x[101:104] = np.nan
    with tempfile.TemporaryDirectory() as d:
        ep, lab = run_synthetic({'GGG': ('USD', 'fixed', x)}, d)
        old, _ = run_synthetic({'GGG': ('USD', 'fixed', x)}, d, fill_counts=True)
    assert len(ep) == 0                          # the carried-forward hours are not samples
    assert len(old) == 1                         # v1.0 counted them (--fill-counts)
    la = lab.set_index('hour')
    assert la['obs'].iloc[101:104].eq(0).all() and la['obs'].iloc[100] == 1
    assert np.allclose(la['dev'].iloc[101:104], -0.015)   # features still see the last price


def test_two_prints_across_a_gap_still_start_an_episode():
    x = noisy(len(IDX), 22)
    x[100], x[103] = 0.985, 0.984                # two prints below -1 % with no price in between
    x[101:103] = np.nan
    with tempfile.TemporaryDirectory() as d:
        ep, _ = run_synthetic({'HHH': ('USD', 'fixed', x)}, d)
    assert len(ep) == 1 and ep.iloc[0]['start'] == IDX[100]
    # the episode is confirmed only by the second print, three hours after the start; features and decisions
    # must not use it before then
    assert pd.Timestamp(ep.iloc[0]['confirmed_at']) == IDX[103]
    from common import known_ts
    assert int(known_ts(ep).iloc[0]) == int(IDX[103].timestamp())


def test_a_normal_start_is_known_one_hour_after_it_begins():
    x = noisy(len(IDX), 24)
    x[100:110] = 0.985                           # no gaps: the second sample is the next hour
    with tempfile.TemporaryDirectory() as d:
        ep, _ = run_synthetic({'JJJ': ('USD', 'fixed', x)}, d)
    from common import known_ts
    assert len(ep) == 1 and pd.Timestamp(ep.iloc[0]['confirmed_at']) == IDX[101]
    assert int(known_ts(ep).iloc[0]) == int(IDX[101].timestamp())


def test_a_deep_print_before_a_gap_holds_no_depth():
    x = noisy(len(IDX), 23)
    x[100:160] = 0.985                           # a mild episode
    x[130] = 0.10                                # one bad print, then three hours without a price (DOLA, Feb 2024)
    x[131:134] = np.nan
    with tempfile.TemporaryDirectory() as d:
        ep, _ = run_synthetic({'III': ('USD', 'fixed', x)}, d)
        old, _ = run_synthetic({'III': ('USD', 'fixed', x)}, d, fill_counts=True)
    assert len(ep) == 1 and ep.iloc[0]['severity'] == 'minor' and abs(ep.iloc[0]['held_min_dev'] + 0.015) < 0.002
    assert old.iloc[0]['severity'] == 'collapse'  # the carried-forward copy made the print look held


def test_episodes_checked_against_dex_and_excluded():
    import compare_sources as cs
    idx = pd.date_range('2025-01-01', periods=600, freq='h', tz='UTC')
    rng = np.random.default_rng(7)
    a = 1 + rng.normal(0, 0.0003, len(idx))
    a[100:104], a[200:203], a[300:304] = 0.97, 0.98, 0.96
    b = 1 + rng.normal(0, 0.0003, len(idx))
    llama = pd.concat([pd.DataFrame({'symbol': sym, 'hour': idx.strftime('%Y-%m-%dT%H:00Z'), 'price': x})
                       for sym, x in (('AAA', a), ('BBB', b))])
    llama['confidence'], llama['source'] = 0.99, 'defillama'
    rows = []                             # DEX rows per asset, quote token and hour, as sql/03 writes them
    for i, t in enumerate(idx):
        # AAA: no DEX trades around hour 300, at peg around 200; at 400-404 a loan inside the hour drags
        # the volume-weighted price to 0.90 while the median trade stays at the peg
        if not 290 <= i <= 320:
            p = 0.97 if 100 <= i < 104 else 1 + rng.normal(0, 0.0003)
            if 400 <= i < 405:
                rows.append(('AAA', 'USDC', t, 5, 0.90, 0.90, p, p))
            else:
                rows.append(('AAA', 'USDC', t, 2, p, p, p, p))
        for q in ('USDC', 'USDT'):        # BBB: a DEX-only dip, seen against both quote tokens
            p = 0.97 if 450 <= i < 455 else 1 + rng.normal(0, 0.0003)
            rows.append(('BBB', q, t, 3, p, p, p, p))
    dex = pd.DataFrame(rows, columns=['symbol', 'quote', 'hour', 'n_trades', 'vwap', 'p25', 'p50', 'p75'])
    dex['hour'] = dex['hour'].dt.strftime('%Y-%m-%dT%H:00Z')
    dex['unit'], dex['source'], dex['n_parties'], dex['volume_quote'] = 'USD', 'dex', dex['n_trades'], 1000.0
    reg = {'chain': 'ethereum', 'multicall3': '0x0', 'assets': [
        {'symbol': sym, 'category': 'synthetic', 'peg': 'USD', 'reference': {'kind': 'fixed', 'value': 1.0}, 'in_scope': True}
        for sym in ('AAA', 'BBB')]}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'reg.json').write_text(json.dumps(reg))
        (d / 'known.json').write_text(json.dumps([{'symbol': 'BBB', 'date': '2025-01-19', 'expect': 'onchain', 'note': 't'}]))
        llama.to_csv(d / 'llama.csv', index=False)
        dex.to_csv(d / 'dex.csv', index=False)
        base = {'prices': [str(d / 'llama.csv')], 'primary': '', 'rates': '', 'registry': str(d / 'reg.json'),
                'rules': str(ROOT / 'config' / 'label_rules.json'), 'known': str(d / 'known.json'), 'out_dir': str(d / 'out')}
        ml.run(type('A', (), base))
        chk, only, kn, plc = cs.run(type('A', (), {'labels': str(d / 'out'), 'dex': str(d / 'dex.csv'), 'rates': '',
                                                   'registry': base['registry'], 'rules': base['rules'], 'known': base['known'],
                                                   'top': 5, 'min_hour_trades': 2}))
        assert list(chk.sort_values('start')['status']) == ['confirmed', 'contradicted', 'unverified']
        assert len(only) == 1 and only.iloc[0]['symbol'] == 'BBB' and only.iloc[0]['start'] == idx[450]   # not the loan hours
        assert len(plc) > 0 and (plc['status'] != 'confirmed').all() and plc['shift_days'].abs().min() >= 7
        hourly = ml.collapse_dex_quotes(pd.read_csv(d / 'dex.csv'))
        h = hourly.set_index(['symbol', 'hour'])
        assert h.loc[('BBB', idx[450].strftime('%Y-%m-%dT%H:00Z')), 'n_trades'] == 6
        assert abs(h.loc[('AAA', idx[401].strftime('%Y-%m-%dT%H:00Z')), 'price'] - 1) < 0.01
        assert bool(kn.iloc[0]['dex_beyond_threshold'])
        excl = pd.read_csv(d / 'out' / 'exclude_contradicted.csv')
        assert len(excl) == 1 and pd.Timestamp(excl.iloc[0]['start']) == idx[200]
        assert '## DEX-only episodes (1)' in (d / 'out' / 'source_check.md').read_text()
        # the robustness run flags it 'excluded': it stays in episodes.csv and its run-up loses its labels
        ml.run(type('A', (), {**base, 'out_dir': str(d / 'out2'), 'exclude_episodes': str(d / 'out' / 'exclude_contradicted.csv')}))
        ep2 = pd.read_csv(d / 'out2' / 'episodes.csv', parse_dates=['start']).fillna({'suspect': ''})
        assert dict(zip(ep2['start'], ep2['suspect']))[idx[200]] == 'excluded' and (ep2['suspect'] == 'excluded').sum() == 1
        lab1 = pd.read_csv(d / 'out' / 'labels_hourly.csv.gz', parse_dates=['hour']).set_index(['symbol', 'hour'])['y24']
        lab2 = pd.read_csv(d / 'out2' / 'labels_hourly.csv.gz', parse_dates=['hour']).set_index(['symbol', 'hour'])['y24']
        assert lab1.loc[('AAA', idx[190])] == 1 and np.isnan(lab2.loc[('AAA', idx[190])])
        assert lab1.loc[('AAA', idx[90])] == 1 and lab2.loc[('AAA', idx[90])] == 1
        cov = pd.read_csv(d / 'out' / 'coverage.csv')
        assert cov['median_value_to_ref'].between(0.999, 1.001).all()


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
