"""Offline tests for make_escalation.py and run_escalation.py on synthetic data.

python tests/test_escalation.py        (needs scikit-learn and scipy)
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
import make_escalation as me  # noqa: E402
import make_labels as ml  # noqa: E402
import run_escalation as rx  # noqa: E402

H = 3600
RULES = json.loads((ROOT / 'config' / 'label_rules.json').read_text())
T0 = int(pd.Timestamp('2024-06-01', tz='UTC').timestamp())
N = 640 * 24
iso = lambda t: pd.Timestamp(int(t), unit='s', tz='UTC').isoformat()


def synthetic(n_assets=24, effect=4.0, seed=0, cut=None):
    """Depeg episodes whose chance of turning severe rises with the leverage of the asset's lending
    exposure (family borrow / collateral) at the start; `effect` = slope of the log-odds in leverage.
    With `cut`, everything after the cut is rewritten (deviations, exposure, ETH, later episodes and
    the outcome of an episode open at the cut) while the past stays as it was."""
    rng = np.random.default_rng(seed)
    ts = T0 + H * np.arange(N)
    assets, labels, feats, episodes = [], [], [], []
    for k in range(n_assets):
        sym, peg = f'A{k:02d}', ('ETH' if k % 4 == 3 else 'USD')
        th = RULES['threshold'][peg]
        assets.append({'symbol': sym, 'peg': peg, 'category': 'synthetic' if peg == 'USD' else 'lst', 'in_scope': True})
        dev = rng.normal(0, 0.05 * th, N)
        lev = np.empty(N)                                   # slowly moving leverage in [0.05, 0.95]
        lev[0] = rng.uniform(0.2, 0.8)
        steps = rng.normal(0, 0.01, N)
        for i in range(1, N):
            lev[i] = np.clip(0.998 * lev[i - 1] + 0.002 * 0.5 + steps[i], 0.05, 0.95)
        starts = np.sort(rng.choice(np.arange(300, N - 300, 24 * 9), size=12, replace=False))
        for s in starts:
            dur = int(rng.integers(3, 48))
            severe = rng.uniform() < 1 / (1 + np.exp(-(-2.0 + effect * (lev[s] - 0.5))))
            off, m = int(rng.integers(2, 12)), int(rng.integers(2, 6))      # the slide starts 2-11 h in
            if severe:
                dur = max(dur, off + m + 1)
            dev[s:s + dur] = -np.minimum(rng.uniform(1.2, 2.0, dur) * th, 0.045)
            if severe:
                dev[s + off:s + off + m] = -rng.uniform(0.06, 0.12, m)
            w = dev[s:s + dur]
            held = me.held_depth(w)
            episodes.append({'symbol': sym, 'category': assets[-1]['category'], 'peg': peg, 'start': iso(ts[s]),
                             'end': iso(ts[s + dur - 1]), 'hours_below': dur, 'min_dev': float(w.min()),
                             'held_min_dev': held, 'severity': ml.severity(held, RULES['severity_tiers']),
                             'ongoing': False, 'suspect': ''})
        labels.append(pd.DataFrame({'hour': pd.to_datetime(ts, unit='s', utc=True).astype(str), 'symbol': sym,
                                    'dev': dev, 'in_episode': 0, 'y24': 0.0, 'y72': 0.0}))
        coll = np.exp(rng.normal(15, 0.3, N))
        f = pd.DataFrame({'symbol': sym, 't': ts, 'time': ''})
        for c in md.EXPOSURE_USD:
            f[c] = coll * rng.uniform(0.5, 1.5)
        for c in md.EXPOSURE_RAW:
            f[c] = rng.uniform(0, 1, N)
        f['family_collateral_usd'] = coll
        f['family_borrow_against_usd'] = lev * coll
        feats.append(f)
    lab, tf, ep = pd.concat(labels, ignore_index=True), pd.concat(feats, ignore_index=True), pd.DataFrame(episodes)
    weth = pd.DataFrame({'symbol': 'WETH', 'ts': ts, 'price': 3000 * np.exp(np.cumsum(rng.normal(0, 0.003, N)))})
    if cut is not None:
        late = pd.to_datetime(lab['hour'], utc=True) > pd.Timestamp(cut, unit='s', tz='UTC')
        lab.loc[late, 'dev'] = lab.loc[late, 'dev'] * 3 - 0.004
        tf.loc[tf['t'] > cut, md.EXPOSURE_USD] *= 0.3                                  # every exposure measure
        tf.loc[tf['t'] > cut, md.EXPOSURE_RAW] = 0.5
        weth.loc[weth['ts'] > cut, 'price'] *= 0.5
        st = pd.to_datetime(ep['start'], utc=True)
        en = pd.to_datetime(ep['end'], utc=True)
        c = pd.Timestamp(cut, unit='s', tz='UTC')
        open_at_cut = (st <= c) & (en > c)
        ep.loc[open_at_cut, ['severity', 'held_min_dev']] = ['collapse', -0.5]
        ep.loc[open_at_cut, 'end'] = (en[open_at_cut] + pd.Timedelta(hours=30)).map(lambda x: x.isoformat())
        extra = ep[st > c].copy()
        extra['start'] = (pd.to_datetime(extra['start'], utc=True) + pd.Timedelta(hours=5)).map(lambda x: x.isoformat())
        ep = pd.concat([ep[st <= c], extra], ignore_index=True)
    return assets, lab, ep, weth, tf


def write_inputs(d, assets, lab, ep, weth, tf):
    d = Path(d)
    (d / 'labels').mkdir(parents=True, exist_ok=True)
    (d / 'registry.json').write_text(json.dumps({'assets': assets}))
    (d / 'wrappers.json').write_text(json.dumps({'pairs': []}))
    lab.to_csv(d / 'labels' / 'labels_hourly.csv.gz', index=False)
    ep.to_csv(d / 'labels' / 'episodes.csv', index=False)
    weth.to_csv(d / 'prices.csv.gz', index=False)
    tf.to_csv(d / 'features.csv.gz', index=False)
    return ['--registry', str(d / 'registry.json'), '--wrappers', str(d / 'wrappers.json'),
            '--labels', str(d / 'labels' / 'labels_hourly.csv.gz'), '--episodes', str(d / 'labels' / 'episodes.csv'),
            '--prices', str(d / 'prices.csv.gz'), '--features', str(d / 'features.csv.gz')]


def build(d, frames, landmarks=(1, 6)):
    argv = write_inputs(d, *frames)
    sys.argv = ['x'] + argv + ['--out', f'{d}/model/escalation.csv.gz', '--landmarks'] + [str(x) for x in landmarks]
    with contextlib.redirect_stdout(io.StringIO()):
        me.main()
    return pd.read_csv(f'{d}/model/escalation.csv.gz'), json.loads(Path(f'{d}/model/escalation_features.json').read_text())


def test_episode_rows_labels_and_exclusions():
    idx = T0 + H * np.arange(200)
    dev = pd.Series(0.0, index=idx)
    dev.iloc[10:14] = [-0.02, -0.03, -0.08, -0.09]        # A: mild first two hours, then severe -> y_severe = 1
    dev.iloc[60:62] = [-0.07, -0.08]                      # B: severe within the first two hours -> decided
    dev.iloc[110:140] = -0.015                            # C: 30 h long, mild
    dev.iloc[180:182] = -0.02                             # D: still open at the end of the data, mild -> censored
    ep = pd.DataFrame({'symbol': 'X', 'start': [iso(idx[i]) for i in (10, 60, 110, 180)],
                       'end': [iso(idx[i]) for i in (13, 61, 139, 181)],
                       'severity': ['major', 'major', 'minor', 'minor'], 'held_min_dev': [-0.08, -0.07, -0.015, -0.02],
                       'ongoing': [False, False, False, True]})
    ep['t0'], ep['t1'] = [idx[i] for i in (10, 60, 110, 180)], [idx[i] for i in (13, 61, 139, 181)]
    r = me.episode_rows(ep, {'X': dev}, {'X': 0.01}, [1, 6], 0.05, 0.5).set_index(['start', 'landmark'])
    a, b, c, d = (iso(idx[i]) for i in (10, 60, 110, 180))
    assert r.loc[(a, 1), 'y_severe'] == 1 and not r.loc[(a, 1), 'decided_severe']
    assert abs(r.loc[(a, 1), 'depth_frac'] - 0.6) < 1e-9 and r.loc[(a, 1), 'ep_hours_below'] == 2
    assert r.loc[(a, 6), 'decided_severe'] and np.isnan(r.loc[(a, 6), 'y_severe'])          # severe by then
    assert r.loc[(b, 1), 'decided_severe'] and np.isnan(r.loc[(b, 1), 'y_severe'])
    assert r.loc[(c, 1), 'y_severe'] == 0 and r.loc[(c, 1), 'y_long'] == 1 and r.loc[(a, 1), 'y_long'] == 0
    assert np.isnan(r.loc[(d, 1), 'y_severe']) and np.isnan(r.loc[(d, 1), 'y_long'])       # open and mild: unknown
    # history: C sees A and B before it (both severe), A sees nothing
    assert r.loc[(c, 1), 'prior_episodes'] == 2 and r.loc[(c, 1), 'prior_severe'] == 2
    assert abs(r.loc[(c, 1), 'prior_severe_share'] - 3 / 7) < 1e-9 and r.loc[(a, 1), 'prior_episodes'] == 0
    assert abs(r.loc[(c, 1), 'hours_since_prior_end'] - (111 - 61)) < 1e-9
    assert abs(r.loc[(c, 1), 'prior_max_depth_frac'] - 1.6) < 1e-9


def test_decision_waits_for_the_sample_that_confirms_the_episode():
    """One print below the threshold, a gap, then the confirming print: the 1-h decision moves to that print."""
    idx = T0 + H * np.arange(100)
    dev = pd.Series(0.0, index=idx)
    dev.iloc[20], dev.iloc[24:30] = -0.02, -0.03          # start at 20; hours 21-23 carry no price of their own
    obs = dev.copy()
    obs.iloc[21:24] = np.nan
    ep = pd.DataFrame({'symbol': 'X', 'start': [iso(idx[20])], 'end': [iso(idx[29])], 'severity': ['minor'],
                       'held_min_dev': [-0.03], 'ongoing': [False]})
    ep['t0'], ep['t1'], ep['tk'] = [idx[20]], [idx[29]], [idx[24]]
    r = me.episode_rows(ep, {'X': dev}, {'X': 0.01}, [1, 6], 0.05, 0.5, dev_obs={'X': obs}).set_index('landmark')
    assert r.loc[1, 'ts'] == idx[24] and r.loc[6, 'ts'] == idx[26]   # not before the episode is known
    without = me.episode_rows(ep.drop(columns='tk'), {'X': dev}, {'X': 0.01}, [1], 0.05, 0.5)
    assert without.loc[0, 'ts'] == idx[21]                          # files without known hours: start + 1 h


def test_held_depth_uses_observed_hours_only():
    """A deep print followed by hours carried forward over a gap holds no depth: the episode is not decided."""
    t0 = T0 + 100 * H
    idx = T0 + H * np.arange(400)
    dev = pd.Series(0.0, index=idx)
    dev.loc[t0] = -0.02                                   # start
    dev.loc[t0 + H] = -0.08                               # one print beyond -5 %
    dev.loc[t0 + 2 * H:t0 + 4 * H] = -0.08                # carried forward over a 3-hour gap
    dev.loc[t0 + 5 * H:t0 + 30 * H] = -0.015
    obs = dev.copy()
    obs.loc[t0 + 2 * H:t0 + 4 * H] = np.nan               # what make_labels marks obs = 0
    ep = pd.DataFrame([{'symbol': 'AAA', 'start': iso(t0), 't0': t0, 't1': t0 + 30 * H, 'severity': 'minor',
                        'held_min_dev': -0.02, 'ongoing': False}])
    args = ({'AAA': dev}, {'AAA': 0.01}, [1, 6], 0.05, 0.5)
    seen = me.episode_rows(ep, *args, dev_obs={'AAA': obs}).set_index('landmark')
    filled = me.episode_rows(ep, *args).set_index('landmark')
    assert not seen.loc[6, 'decided_severe'] and seen.loc[6, 'y_severe'] == 0
    assert filled.loc[6, 'decided_severe']              # counting the carried-forward hours would decide it
    assert seen.loc[6, 'depth_frac'] == filled.loc[6, 'depth_frac'] == 0.08 / 0.05   # features see the print


def test_escalation_table_has_no_look_ahead():
    cut = T0 + 300 * 24 * H
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a, groups = build(d1, synthetic(n_assets=6))
        b, _ = build(d2, synthetic(n_assets=6, cut=cut))
        cols = groups['EP'] + groups['P'] + groups['E']
        pa = a[a['ts'] <= cut].set_index(['episode_id', 'landmark'])[cols].sort_index()
        pb = b[b['ts'] <= cut].set_index(['episode_id', 'landmark'])[cols].sort_index()
        assert pa.index.equals(pb.index) and len(pa) > 30
        same = np.isclose(pa.to_numpy(float), pb.to_numpy(float), rtol=1e-5, atol=1e-6, equal_nan=True)
        bad = [c for c, ok in zip(cols, same.all(axis=0)) if not ok]
        assert not bad, bad
        assert not a.loc[a['ts'] > cut, 'pre_z_mean_168h'].reset_index(drop=True).equals(
            b.loc[b['ts'] > cut, 'pre_z_mean_168h'].reset_index(drop=True))                  # the rewrite is real


def test_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = (rng.uniform(size=300) < 0.2).astype(float)
        s = np.round(rng.normal(size=300) + y, 1)          # many ties
        assert abs(rx.ap_score(y, s) - average_precision_score(y, s)) < 1e-12
        assert abs(rx.auc_score(y, s) - roc_auc_score(y, s)) < 1e-12


def test_clustered_standard_errors():
    rng = np.random.default_rng(2)
    n = 3000
    X = rng.normal(size=(n, 2))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-1 + 0.8 * X[:, 0])))).astype(float)
    b1, V1, conv, _ = rx.logit_clustered(X, y, np.arange(n))
    assert conv and abs(b1[1] - 0.8) < 4 * np.sqrt(V1[1, 1]) and abs(b1[2]) < 4 * np.sqrt(V1[2, 2])
    p = 1 / (1 + np.exp(-(np.column_stack([np.ones(n), X]) @ b1)))
    naive = np.sqrt(np.diag(np.linalg.inv((np.column_stack([np.ones(n), X]) * (p * (1 - p))[:, None]).T @ np.column_stack([np.ones(n), X]))))
    assert np.allclose(np.sqrt(np.diag(V1)), naive, rtol=0.1)          # well specified: robust ~ model-based
    rep = 5                                                            # five copies of every row, clustered: same SEs
    b2, V2, _, _ = rx.logit_clustered(np.repeat(X, rep, axis=0), np.repeat(y, rep), np.repeat(np.arange(n), rep))
    assert np.allclose(b1, b2, atol=1e-6) and np.allclose(np.sqrt(np.diag(V2)), np.sqrt(np.diag(V1)), rtol=0.02)
    # a rare feature that only ever comes with y = 0 (quasi-separation) must not throw the others off
    Xs = np.column_stack([X, np.where(np.arange(n) < 20, 3.0, 0.0)])
    ys = y.copy()
    ys[:20] = 0
    b3, V3, conv3, _ = rx.logit_clustered(Xs, ys, np.arange(n) // 10)
    assert conv3 and abs(b3[1] - b1[1]) < 0.05 and b3[3] < -1 and np.isfinite(np.diag(V3)).all()


def test_bootstrap_sets_match_single_block():
    rng = np.random.default_rng(5)
    n = 1200
    X = rng.normal(size=(n, 3))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-1 + 0.6 * X[:, 0])))).astype(float)
    g = np.arange(n) // 20
    beta, V, _, _ = rx.logit_clustered(X, y, g)
    p1, w1, d1 = rx.bootstrap_t(X, y, g, beta, V, [2, 3], 40)
    p2, w2, d2 = rx.bootstrap_t_sets(X, y, g, beta, V, {'b': [2, 3], 'a': [1], 'none': []}, 40)
    assert d1 == d2 == 40 and np.allclose(p1, p2) and w1 == w2['b'] and set(w2) == {'a', 'b'}
    assert w2['a'] < 0.1 < w2['b']                                     # x0 matters, x1 and x2 do not


def test_exposure_effect_is_found_and_null_is_not():
    for effect, expect in ((5.0, True), (0.0, False)):
        with tempfile.TemporaryDirectory() as d:
            table, _ = build(d, synthetic(effect=effect, seed=3), landmarks=(1,))
            sys.argv = ['x', '--data', f'{d}/model/escalation.csv.gz', '--out', f'{d}/model', '--outcomes', 'severe',
                        '--bootstrap', '100', '--inference-bootstrap', '199',
                        '--models', 'prior', 'depth', 'asset_hist', 'logit_core', 'logit_coreE']
            with contextlib.redirect_stdout(io.StringIO()):
                rx.main()
            r = json.loads(Path(f'{d}/model/escalation_results.json').read_text())[0]
            inf = r['inference']['all']
            lev = next(c for c in inf['coef'] if c['feature'] == 'family_borrow_to_collateral')
            assert inf['bootstrap_draws'] > 150 and all(0 < c['p_boot'] <= 1 for c in inf['coef'])
            if expect:
                assert inf['wald_E']['p'] < 0.01 and lev['odds_ratio'] > 1.5 and lev['p'] < 0.01, inf
                assert lev['p_boot'] < 0.05 and inf['wald_E']['p_boot'] < 0.05, inf
                assert r['protocols']['loao']['pairs']['logit_coreE - logit_core']['ap'] > 0.02
            else:
                assert inf['wald_E']['p'] > 0.01 and inf['wald_E']['p_boot'] > 0.05, inf['wald_E']
            assert abs(r['protocols']['time']['models']['prior']['roc_auc'] - 0.5) < 1e-12
            assert r['n'] == table.loc[table['landmark'] == 1, 'y_severe'].notna().sum()
            assert '## Severe, 1 h after the start' in Path(f'{d}/model/escalation_results.md').read_text()


def test_runner_end_to_end_with_trees():
    with tempfile.TemporaryDirectory() as d:
        table, groups = build(d, synthetic(n_assets=10, seed=4), landmarks=(1, 6))
        assert {'depth_frac', 'prior_severe_share', 'z_now', 'has_exposure'} <= set(table.columns)
        assert set(table['split']) == {'train', 'valid', 'test'}
        sys.argv = ['x', '--data', f'{d}/model/escalation.csv.gz', '--out', f'{d}/model', '--bootstrap', '50',
                    '--inference-bootstrap', '20']
        with contextlib.redirect_stdout(io.StringIO()):
            rx.main()
        res = json.loads(Path(f'{d}/model/escalation_results.json').read_text())
        assert {(r['outcome'], r['landmark']) for r in res} == {('severe', 1), ('severe', 6), ('long', 1), ('long', 6)}
        for r in res:
            for p in ('time', 'loao'):
                assert set(r['protocols'][p]['models']) == set(rx.MODELS)
        p = pd.read_csv(f'{d}/model/escalation_preds_severe_1h.csv.gz')
        assert set(p['protocol']) == {'time', 'loao'} and p['score'].notna().all()
        assert p[p['protocol'] == 'loao'].groupby('model').size().nunique() == 1   # every model scores every episode


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
