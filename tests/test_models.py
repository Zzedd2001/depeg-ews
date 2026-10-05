"""Offline tests for make_dataset.py and run_baselines.py on synthetic data.

python tests/test_models.py        (needs scikit-learn for the baseline test)
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import make_dataset as md  # noqa: E402
import compare_preds as cp  # noqa: E402
import run_baselines as rb  # noqa: E402
from common import epoch_seconds, not_suspect  # noqa: E402

H = 3600
T0 = int(pd.Timestamp('2025-01-01', tz='UTC').timestamp())
N = 365 * 24
ASSETS = {'USDe': 0.01, 'sUSDe': 0.01, 'wstETH': 0.02}       # label thresholds (USD 1 %, ETH 2 %)


def synthetic(seed=0, perturb_after=None):
    """Hourly deviations with episodes; each episode is preceded by a 12-hour slide toward the threshold."""
    rng = np.random.default_rng(seed)
    ts = T0 + H * np.arange(N)
    labels, episodes, feats = [], [], []
    for k, (sym, thr) in enumerate(ASSETS.items()):
        dev = rng.normal(0, thr * 0.05, N)
        starts = np.sort(rng.choice(np.arange(200, N - 200, 24), size=14, replace=False))
        in_ep = np.zeros(N, bool)
        for s in starts:
            dev[s - 12:s] -= np.linspace(0, 0.9 * thr, 12)          # the warning sign
            dev[s:s + 6] -= 1.5 * thr
            in_ep[s:s + 6] = True
            episodes.append((sym, pd.Timestamp(ts[s], unit='s', tz='UTC').isoformat(), ''))
        y = {}
        for h in (24, 72):
            yy = np.zeros(N)
            for s in starts:
                yy[max(0, s - h):s] = 1
            yy[in_ep] = np.nan
            y[h] = yy
        df = pd.DataFrame({'hour': pd.to_datetime(ts, unit='s', utc=True).astype(str), 'symbol': sym, 'dev': dev,
                           'in_episode': in_ep.astype(int), 'y24': y[24], 'y72': y[72]})
        labels.append(df)
        exp = np.exp(rng.normal(16, 0.5, N)) if sym != 'wstETH' else np.full(N, np.nan)
        f = pd.DataFrame({'symbol': sym, 't': ts, 'time': ''})
        for c in md.EXPOSURE_USD:
            f[c] = exp * (1 + 0.1 * k)
        for c in md.EXPOSURE_RAW:
            f[c] = rng.uniform(0, 1, N)
        feats.append(f)
    lab, tf = pd.concat(labels, ignore_index=True), pd.concat(feats, ignore_index=True)
    weth = pd.DataFrame({'symbol': 'WETH', 'ts': ts + rng.integers(-600, 600, N), 'price': 3000 * np.exp(np.cumsum(rng.normal(0, 0.003, N)))})
    ep = pd.DataFrame(episodes, columns=['symbol', 'start', 'suspect'])
    if perturb_after is not None:                       # rewrite everything after the cut, keep the past as it was
        late = epoch_seconds(lab['hour']) > perturb_after
        lab.loc[late, 'dev'] = lab.loc[late, 'dev'] * -3 + 0.05
        tf.loc[tf['t'] > perturb_after, md.EXPOSURE_USD] = 1.0
        tf.loc[tf['t'] > perturb_after, md.EXPOSURE_RAW] = 0.5      # every exposure measure
        weth.loc[weth['ts'] > perturb_after, 'price'] *= 0.5
        extra = pd.DataFrame([(s, pd.Timestamp(perturb_after + H, unit='s', tz='UTC').isoformat(), '') for s in ASSETS],
                             columns=ep.columns)
        ep = pd.concat([ep, extra], ignore_index=True)
    return lab, ep, weth, tf


def write_inputs(d, lab, ep, weth, tf):
    d = Path(d)
    (d / 'labels').mkdir(parents=True, exist_ok=True)
    lab.to_csv(d / 'labels' / 'labels_hourly.csv.gz', index=False)
    ep.to_csv(d / 'labels' / 'episodes.csv', index=False)
    weth.to_csv(d / 'prices.csv.gz', index=False)
    tf.to_csv(d / 'features.csv.gz', index=False)
    return ['--labels', str(d / 'labels' / 'labels_hourly.csv.gz'), '--episodes', str(d / 'labels' / 'episodes.csv'),
            '--prices', str(d / 'prices.csv.gz'), '--features', str(d / 'features.csv.gz')]


def build(d, *frames):
    argv = write_inputs(d, *frames)
    sys.argv = ['x'] + argv + ['--out', f'{d}/model/dataset.pkl.gz']
    with contextlib.redirect_stdout(io.StringIO()):
        md.main()
    return pd.read_pickle(f'{d}/model/dataset.pkl.gz'), json.loads(Path(f'{d}/model/dataset_features.json').read_text())


def test_dataset_has_no_look_ahead():
    cut = T0 + 200 * 24 * H
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a, groups = build(d1, *synthetic())
        b, _ = build(d2, *synthetic(perturb_after=cut))
        cols = groups['P'] + groups['E']
        pa = a[a['ts'] <= cut].set_index(['ts', 'symbol'])[cols].sort_index()
        pb = b[b['ts'] <= cut].set_index(['ts', 'symbol'])[cols].sort_index()
        assert pa.index.equals(pb.index) and len(pa) > 10000
        same = (pa == pb) | (pa.isna() & pb.isna())
        bad = [c for c in cols if not same[c].all()]
        assert not bad, bad                                        # nothing at or before the cut moved
        after = a[a['ts'] > cut + 200 * H].set_index(['ts', 'symbol'])['z_now']
        assert not after.equals(b[b['ts'] > cut + 200 * H].set_index(['ts', 'symbol'])['z_now'])   # the perturbation is real


def test_dataset_features_and_splits():
    with tempfile.TemporaryDirectory() as d:
        frames = synthetic()
        a, groups = build(d, *frames)
        assert set(a['split']) == {'train', 'valid', 'test'} and a['ts'].is_monotonic_increasing
        assert a.loc[a['ts'] == int(pd.Timestamp('2025-03-31T23:00Z').timestamp()), 'split'].eq('train').all()
        assert a.loc[a['ts'] == int(pd.Timestamp('2025-04-01T00:00Z').timestamp()), 'split'].eq('valid').all()
        lab = frames[0]
        row = a[(a['symbol'] == 'wstETH')].iloc[500]
        w = lab[lab['symbol'] == 'wstETH']
        dev = w.set_index(epoch_seconds(w['hour']))['dev']
        assert abs(row['z_now'] - dev[row['ts']] / 0.02) < 1e-5                   # threshold units, ETH = 2 %
        assert abs(row['z_min_24h'] - (dev.loc[row['ts'] - 23 * H:row['ts']] / 0.02).min()) < 1e-5
        su = a[a['symbol'] == 'sUSDe'].set_index('ts')['und_z_now']
        us = a[a['symbol'] == 'USDe'].set_index('ts')['z_now']
        common = su.index.intersection(us.index)[:1000]
        assert np.allclose(su[common], us[common])                                # wrapper sees its underlying
        assert a.loc[a['symbol'] == 'USDe', 'und_z_now'].isna().all()
        # an episode starting at s counts from s + 1 h
        ep = frames[1]
        s = int(pd.Timestamp(ep[ep['symbol'] == 'USDe']['start'].iloc[0]).timestamp())
        u = a[a['symbol'] == 'USDe'].set_index('ts')
        before = u.loc[u.index[u.index < s + H].max(), 'episodes_30d']
        assert before == 0
        assert {'log_family_collateral_usd', 'dlog_mm_borrow_against_usd_24h', 'has_exposure'} <= set(groups['E'])
        assert a.loc[a['symbol'] == 'wstETH', 'has_exposure'].eq(0).all() and a.loc[a['symbol'] == 'USDe', 'has_exposure'].eq(1).all()


def test_flagged_episodes_are_ignored():
    """Episodes make_labels flagged as bad prints ('snapback', ...) must not count as history."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        lab, ep, weth, tf = synthetic()
        a, _ = build(d1, lab, ep, weth, tf)
        flagged = pd.DataFrame([('USDe', pd.Timestamp(T0 + 100 * 24 * H, unit='s', tz='UTC').isoformat(), 'snapback')],
                               columns=ep.columns)
        b, _ = build(d2, lab, pd.concat([ep, flagged], ignore_index=True), weth, tf)
        cols = ['episodes_30d', 'episodes_365d', 'hours_since_episode', 'mkt_episodes_24h']
        pd.testing.assert_frame_equal(a[cols], b[cols])
    keep = not_suspect(pd.DataFrame({'suspect': [np.nan, '', 'snapback', 'same_hour_cluster', True, False, 'True']}))
    assert list(keep.index) == [0, 1, 5]


def test_an_episode_counts_from_the_sample_that_confirms_it():
    """A gap in the data can delay an episode's second sample; history and market features must wait for it."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        lab, ep, weth, tf = synthetic()
        a, _ = build(d1, lab, ep, weth, tf)
        late = ep.copy()
        late['confirmed_at'] = late['start']                      # known one hour after the start, as before
        i = late.index[late['symbol'] == 'USDe'][0]
        s = int(pd.Timestamp(late.loc[i, 'start']).timestamp())
        late.loc[i, 'confirmed_at'] = pd.Timestamp(s + 4 * H, unit='s', tz='UTC').isoformat()   # ... except this one
        b, _ = build(d2, lab, late, weth, tf)
    key = ['ts', 'symbol']
    m = a.set_index(key)['mkt_episodes_24h'].to_frame('a').join(b.set_index(key)['mkt_episodes_24h'].rename('b'))
    other = m[m.index.get_level_values('symbol') != 'USDe']
    t = other.index.get_level_values('ts')
    gap = other[(t >= s + H) & (t < s + 4 * H)]
    after = other[(t >= s + 4 * H) & (t < s + 24 * H)]
    assert len(gap) and (gap['a'] - gap['b'] == 1).all()          # not counted before its confirming sample
    assert len(after) and (after['a'] == after['b']).all()        # counted from then on


def test_event_metrics():
    df = pd.DataFrame({'symbol': 'A', 'ts': T0 + H * np.arange(100), 'y24': 0.0, 'has_exposure': 1.0})
    df.loc[50:59, 'y24'] = 1.0                  # episode starts at hour 60
    score = np.zeros(100)
    score[54] = 1.0                             # first alert 6 h before the start
    eps = pd.DataFrame({'symbol': ['A'], 'ts': [T0 + 60 * H]})
    out = rb.evaluate(df, score, 'y24', 24, eps.assign(severity='major'), {0.01: 0.5})
    b = out['budget_0.01']
    assert b['episodes'] == 1 and b['event_recall'] == 1.0 and b['median_lead_h'] == 6.0
    assert b['severe_episodes'] == 1 and b['severe_recall'] == 1.0
    assert out['rate_0.05']['alert_rate'] == 0.05 and out['rate_0.05']['episodes'] == 1
    assert b['alert_precision'] == 1.0 and abs(b['alert_rate'] - 0.01) < 1e-12
    out = rb.evaluate(df, score, 'y24', 24, eps, {0.01: 2.0})
    assert out['budget_0.01']['event_recall'] == 0.0 and out['budget_0.01']['median_lead_h'] is None


def test_baselines_end_to_end():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print('skip (scikit-learn not installed)')
        return
    with tempfile.TemporaryDirectory() as d:
        build(d, *synthetic())
        sys.argv = ['x', '--data', f'{d}/model/dataset.pkl.gz', '--episodes', f'{d}/labels/episodes.csv', '--out', f'{d}/model',
                    '--max-iter', '30', '--importance']
        with contextlib.redirect_stdout(io.StringIO()):
            rb.main()
        res = json.loads(Path(f'{d}/model/results.json').read_text())
        assert {(r['model'], r['horizon']) for r in res} == {(m, h) for m in rb.MODELS for h in (24, 72)}
        get = lambda m, h: next(r for r in res if r['model'] == m and r['horizon'] == h)
        assert get('rule_dev', 24)['test']['ap'] > 3 * get('rule_dev', 24)['test']['prevalence']    # the slide is a signal
        assert get('hgb_P', 24)['test']['ap'] > get('base_rate', 24)['test']['ap']
        assert get('hgb_PE', 24)['importance'] and len(get('hgb_PE', 24)['best_iter']) == 5     # one per seed
        assert min(get('hgb_PE', 24)['best_iter']) >= 10
        sets = rb.feature_sets(json.loads(Path(f'{d}/model/dataset_features.json').read_text()))
        assert not any(c.startswith('log_') for c in sets['hgb_PEd']) and any(c.startswith('dlog_') for c in sets['hgb_PEd'])
        assert get('hgb_E', 24)['test']['ap_unexposed'] is not None                # wstETH hours have no exposure
        assert 0 < get('hgb_P', 72)['test']['budget_0.05']['event_recall'] <= 1
        md_text = Path(f'{d}/model/results.md').read_text()
        assert '## 24 h horizon' in md_text and '| hgb_PE |' in md_text
        p = pd.read_csv(f'{d}/model/preds_hgb_PE_24h.csv.gz')
        assert set(p.columns) == {'symbol', 'ts', 'y', 'has_exposure', 'score'} and p['ts'].min() >= int(pd.Timestamp('2025-10-01', tz='UTC').timestamp())
        # re-running one model replaces its entries instead of duplicating them
        sys.argv = ['x', '--data', f'{d}/model/dataset.pkl.gz', '--episodes', f'{d}/labels/episodes.csv', '--out', f'{d}/model',
                    '--models', 'rule_dev', '--horizons', '24']
        with contextlib.redirect_stdout(io.StringIO()):
            rb.main()
        assert len(json.loads(Path(f'{d}/model/results.json').read_text())) == len(res)
        # paired comparison of two models' test predictions
        sys.argv = ['x', '--dir', f'{d}/model', '--horizons', '24', '--pairs', 'hgb_PE:hgb_P', '--bootstrap', '50']
        with contextlib.redirect_stdout(io.StringIO()):
            cp.main()
        c = json.loads(Path(f'{d}/model/compare_preds.json').read_text())[0]['subsets']
        assert set(c) == {'all', 'exposed', 'unexposed'} and c['all']['draws'] == 50
        assert abs(c['all']['ap_b'] - get('hgb_P', 24)['test']['ap']) < 1e-9
        assert abs(c['exposed']['ap_a'] - get('hgb_PE', 24)['test']['ap_exposed']) < 1e-9


def test_tree_scores_average_over_seeds():
    """The tree scores are the mean of the single-seed fits, each with its own iteration count."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print('skip (scikit-learn not installed)')
        return
    rng = np.random.default_rng(7)
    frames = []
    for n in (600, 300, 300):
        x = rng.normal(size=(n, 3))
        frames.append(pd.DataFrame({'a': x[:, 0], 'b': x[:, 1], 'c': x[:, 2],
                                    'y': (x[:, 0] + rng.normal(size=n) > 1.5).astype(float)}))
    tr, va, te = frames
    sv, st, info = rb.fit_hgb_seeds(tr, va, te, 'y', ['a', 'b', 'c'], 30, [0, 1])
    one = [rb.fit_hgb(tr, va, te, 'y', ['a', 'b', 'c'], 30, seed=s) for s in (0, 1)]
    assert np.allclose(sv, (one[0][0] + one[1][0]) / 2) and np.allclose(st, (one[0][1] + one[1][1]) / 2)
    assert info['best_iter'] == [one[0][2]['best_iter'], one[1][2]['best_iter']]


def test_winsorising_keeps_rare_features():
    """Winsor bounds must not turn a feature that is nonzero in under 1 % of rows into a constant."""
    from common import winsor_bounds
    rare = np.zeros(1000)
    rare[:8] = [3.0, 2.0, 5.0, 1.0, 4.0, 2.5, 6.0, 1.5]           # 0.8 % nonzero, like a seed-candidate link
    common_ = np.linspace(0, 1, 1000)
    frame = pd.DataFrame({'rare': rare, 'common': common_})
    lo, hi = winsor_bounds(frame, 0.01, 0.99)
    clipped = frame.clip(lo, hi, axis=1)
    assert clipped['rare'].std() > 0 and clipped['rare'].max() == 6.0        # kept, at full range
    assert abs(hi['common'] - np.quantile(common_, 0.99)) < 1e-12          # ordinary columns are winsorised


def test_weighted_ap_matches_expanded_data():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(5)
    for _ in range(10):
        y = (rng.uniform(size=400) < 0.1).astype(float)
        s = np.round(rng.normal(size=400) + 2 * y, 1)
        w = rng.integers(0, 4, 400)
        rows = np.repeat(np.arange(400), w)
        assert abs(cp.SortedAP(y, s)(w.astype(float)) - average_precision_score(y[rows], s[rows])) < 1e-12


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
