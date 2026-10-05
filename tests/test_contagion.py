"""Offline tests for make_contagion.py and run_contagion.py on synthetic data.

python tests/test_contagion.py        (needs scikit-learn and scipy)
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
import make_contagion as mc  # noqa: E402
import make_dataset as md  # noqa: E402
import make_labels as ml  # noqa: E402
import run_contagion as rc  # noqa: E402

H = 3600
D = 86400
RULES = json.loads((ROOT / 'config' / 'label_rules.json').read_text())
T0 = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp())
N = 912 * 24
SYMS = [f'A{k:02d}' for k in range(12)]
GROUP = {**{s: 'V1' for s in SYMS[0:4]}, **{s: 'V2' for s in SYMS[4:8]}}          # shared vaults
FAMILY = {('A11', 'A10')}                                                          # A11 wraps A10
iso = lambda t: pd.Timestamp(int(t), unit='s', tz='UTC').isoformat()


def linked(a, b):
    return (GROUP.get(a) is not None and GROUP.get(a) == GROUP.get(b)) or (a, b) in FAMILY or (b, a) in FAMILY


def synthetic(seed=0, p_link=0.7, p_other=0.04, cut=None):
    """Severe seed depegs; assets that share a vault (or a family) with the seed follow with p_link,
    the others with p_other. With `cut`, everything after it is rewritten."""
    rng = np.random.default_rng(seed)
    ts = T0 + H * np.arange(N)
    dev = {s: rng.normal(0, 0.0005, N) for s in SYMS}
    episodes = []

    def put(sym, h, depth_path):
        dev[sym][h:h + len(depth_path)] = depth_path
        w = dev[sym][h:h + len(depth_path)]
        hd = float(np.min(np.maximum(w[:-1], w[1:])))
        last = h + int(np.flatnonzero(w <= -0.005).max())
        episodes.append({'symbol': sym, 'category': 'synthetic' if int(sym[1:]) < 6 else 'cdp', 'peg': 'USD',
                         'start': iso(ts[h]), 'end': iso(ts[last]), 'hours_below': int((w <= -0.01).sum()),
                         'min_dev': float(w.min()), 'held_min_dev': hd, 'severity': ml.severity(hd, RULES['severity_tiers']),
                         'ongoing': False, 'suspect': '', 'terminal': False})
    starts = np.sort(rng.choice(np.arange(14 * 24, N - 20 * 24, 24 * 8), size=95, replace=False))
    starts = starts[np.r_[True, np.diff(starts) >= 8 * 24]]
    for k, h0 in enumerate(starts):
        s = SYMS[rng.integers(len(SYMS))]
        deep = -0.25 if k % 9 == 0 else -0.08
        put(s, h0, np.r_[[-0.015, -0.015], [deep] * 3])
        for j in SYMS:
            if j != s and rng.uniform() < (p_link if linked(s, j) else p_other):
                put(j, h0 + int(rng.integers(10, 60)), np.full(3, -0.015))
    rows = [pd.DataFrame({'hour': pd.to_datetime(ts, unit='s', utc=True).astype(str), 'symbol': s, 'dev': dev[s],
                          'in_episode': 0, 'y24': 0.0, 'y72': 0.0}) for s in SYMS]
    lab, ep = pd.concat(rows, ignore_index=True), pd.DataFrame(episodes)
    # exposure graph: one market per asset, group vaults fund their group's markets
    days = np.arange(int(pd.Timestamp('2024-01-06', tz='UTC').timestamp()), ts[-1], D)
    e = []
    for t in days:
        for k, s in enumerate(SYMS):
            m = f'mm:{k:012d}'
            vault = {'V1': 'mv:000000000001', 'V2': 'mv:000000000002'}.get(GROUP.get(s), f'mv:1000000000{k:02d}')
            e += [(t, f'tok:{s}', m, 'collateral', 1e6), (t, m, 'tok:USDC', 'borrow', 5e5), (t, vault, m, 'allocation', 2e5)]
    edges = pd.DataFrame(e, columns=['t', 'src', 'dst', 'etype', 'usd'])
    edges['time'] = ''
    static = pd.DataFrame([('tok:A11', 'tok:A10', 'wrapper')], columns=['src', 'dst', 'etype'])
    tf = pd.concat([pd.DataFrame({'symbol': s, 't': ts, 'time': '', **{c: rng.uniform(1e5, 1e6, N) for c in md.EXPOSURE_USD + md.EXPOSURE_RAW}})
                    for s in SYMS], ignore_index=True)
    weth = pd.DataFrame({'symbol': 'WETH', 'ts': ts, 'price': 3000 * np.exp(np.cumsum(rng.normal(0, 0.003, N)))})
    if cut is not None:
        c = pd.Timestamp(cut, unit='s', tz='UTC')
        late = pd.to_datetime(lab['hour'], utc=True) > c
        lab.loc[late, 'dev'] = lab.loc[late, 'dev'] * 3 - 0.004
        edges.loc[edges['t'] > cut, 'usd'] *= 0.3
        tf.loc[tf['t'] > cut, md.EXPOSURE_USD] *= 0.5                                  # every exposure measure
        tf.loc[tf['t'] > cut, md.EXPOSURE_RAW] = 0.5
        weth.loc[weth['ts'] > cut, 'price'] *= 0.5
        st, en = pd.to_datetime(ep['start'], utc=True), pd.to_datetime(ep['end'], utc=True)
        open_at_cut = (st <= c) & (en > c)
        ep.loc[open_at_cut, 'end'] = (en[open_at_cut] + pd.Timedelta(hours=30)).map(lambda x: x.isoformat())
        later = ep[st > c].copy()
        later['start'] = (pd.to_datetime(later['start'], utc=True) + pd.Timedelta(hours=7)).map(lambda x: x.isoformat())
        ep = pd.concat([ep[st <= c], later.iloc[::2]], ignore_index=True)
    return lab, ep, edges, static, tf, weth


def write_inputs(d, lab, ep, edges, static, tf, weth):
    d = Path(d)
    for sub in ('labels', 'graph', 'model'):
        (d / sub).mkdir(parents=True, exist_ok=True)
    reg = [{'symbol': s, 'peg': 'USD', 'category': 'synthetic' if int(s[1:]) < 6 else 'cdp', 'in_scope': True} for s in SYMS]
    (d / 'registry.json').write_text(json.dumps({'assets': reg}))
    (d / 'wrappers.json').write_text(json.dumps({'pairs': [['A11', 'A10']]}))
    lab.to_csv(d / 'labels' / 'labels_hourly.csv.gz', index=False)
    ep.to_csv(d / 'labels' / 'episodes.csv', index=False)
    edges.to_csv(d / 'graph' / 'edges.csv.gz', index=False)
    static.to_csv(d / 'graph' / 'edges_static.csv', index=False)
    tf.to_csv(d / 'features.csv.gz', index=False)
    weth.to_csv(d / 'prices.csv.gz', index=False)
    return ['--registry', str(d / 'registry.json'), '--wrappers', str(d / 'wrappers.json'),
            '--labels', str(d / 'labels' / 'labels_hourly.csv.gz'), '--episodes', str(d / 'labels' / 'episodes.csv'),
            '--prices', str(d / 'prices.csv.gz'), '--features', str(d / 'features.csv.gz'), '--graph', str(d / 'graph')]


def build(d, frames, seeds='severe'):
    sys.argv = ['x'] + write_inputs(d, *frames) + ['--seeds', seeds, '--out', f'{d}/model/contagion_{seeds}.csv.gz']
    with contextlib.redirect_stdout(io.StringIO()):
        mc.main()
    return pd.read_csv(f'{d}/model/contagion_{seeds}.csv.gz'), json.loads(Path(f'{d}/model/contagion_features.json').read_text())


def test_seeds_candidates_and_labels():
    idx = T0 + H * np.arange(400)
    flat = lambda: pd.Series(0.0, index=idx)
    dev = {s: flat() for s in ('S', 'B', 'C', 'X', 'Y')}
    dev['S'].iloc[100:105] = [-0.02, -0.03, -0.06, -0.07, -0.01]         # severe from hour 103
    dev['B'].iloc[90:96] = -0.02                                          # open at tau: B started at 90, ended 95 (< 24 h ago)
    dev['C'].iloc[130:133] = -0.02                                        # follows 27 h after tau
    dev['X'].iloc[:] = np.nan
    dev['X'].iloc[:60] = 0.0                                              # no price in the 24 h before tau
    ep = pd.DataFrame([('S', 100, 104, 'major', False), ('B', 90, 95, 'minor', False), ('C', 130, 132, 'minor', False),
                       ('Y', 20, 25, 'collapse', True)], columns=['symbol', 'i0', 'i1', 'severity', 'terminal'])
    ep['t0'], ep['t1'] = idx[ep['i0']], idx[ep['i1']]
    ep['start'] = ep['t0'].map(iso)
    both = mc.seed_table(ep, dev, 'severe', T0, 0.05)              # Y's collapse is a seed too (no held pair: start + 1 h)
    assert list(both['seed']) == ['Y', 'S'] and list(both['tau']) == [idx[21], idx[103]] and list(both['cluster']) == [0, 1]
    seeds = both[both['seed'] == 'S']
    pairs = mc.candidate_rows(seeds, ep, dev, ['B', 'C', 'S', 'X', 'Y'], data_end=idx[-1]).set_index('cand')
    assert list(pairs.index) == ['C']                                      # B open, S itself, X no price, Y dead
    assert pairs.loc['C', 'y72'] == 1 and pairs.loc['C', 'y168'] == 1
    short = mc.candidate_rows(seeds, ep, dev, ['C'], data_end=idx[103] + 100 * H).set_index('cand')
    assert short.loc['C', 'y72'] == 1 and np.isnan(short.loc['C', 'y168'])          # the week runs past the data
    allseeds = mc.seed_table(ep[ep['symbol'] != 'Y'], dev, 'all', T0, 0.05)
    assert list(allseeds['tau']) == list(idx[[91, 101, 131]]) and list(allseeds['cluster']) == [0, 0, 0]
    # a seed whose second sample came after a gap is decided when that sample confirms it
    late = ep[ep['symbol'] != 'Y'].assign(tk=lambda x: x['t0'] + H)
    late.loc[late['symbol'] == 'B', 'tk'] = idx[94]
    later = mc.seed_table(late, dev, 'all', T0, 0.05).set_index('seed')
    assert later.loc['B', 'tau'] == idx[94] and later.loc['S', 'tau'] == idx[101]


def test_candidates_dated_or_known():
    """An asset whose episode starts at the decision hour but is confirmed an hour later is left out under the
    default rule (episodes as dated by make_labels) and kept, as a non-follower, with known_only."""
    tau = T0 + 10 * D
    seeds = pd.DataFrame([{'seed_id': 'S@x', 'seed': 'S', 'seed_severity': 'major', 'seed_t0': tau - D,
                           'tau': tau, 'cluster': 0}])
    ep = pd.DataFrame([
        {'symbol': 'S', 't0': tau - D, 't1': tau + D, 'tk': tau - D + H, 'terminal': False},
        {'symbol': 'A', 't0': tau, 't1': tau + 10 * H, 'tk': tau + H, 'terminal': False},          # starts at tau
        {'symbol': 'B', 't0': tau - 5 * D, 't1': tau - 4 * D, 'tk': tau - 5 * D + H, 'terminal': False},  # long over
        {'symbol': 'C', 't0': tau - 3 * H, 't1': tau + 5 * H, 'tk': tau - 2 * H, 'terminal': False},  # known, open
    ])
    hours = np.arange(tau - 3 * D, tau + 8 * D + H, H)                         # the 7-day window is observed
    dev = {s: pd.Series(-0.001, index=hours) for s in 'SABC'}
    dated = mc.candidate_rows(seeds, ep, dev, list('SABC'), int(hours[-1]))
    known = mc.candidate_rows(seeds, ep, dev, list('SABC'), int(hours[-1]), known_only=True)
    assert sorted(dated['cand']) == ['B']
    assert sorted(known['cand']) == ['A', 'B'] and known.set_index('cand').loc['A', 'y168'] == 0.0


def test_a_missing_family_file_stops_the_build():
    with tempfile.TemporaryDirectory() as d:
        args = write_inputs(d, *synthetic(seed=3))
        (Path(d) / 'graph' / 'edges_static.csv').unlink()
        sys.argv = ['x'] + args + ['--out', f'{d}/model/contagion_severe.csv.gz']
        try:
            mc.main()
        except SystemExit as e:
            assert 'edges_static.csv' in str(e)
        else:
            raise AssertionError('make_contagion ran without the family file')


def test_graph_links_by_hand():
    t = 1_000_000
    nan = np.nan
    edges = pd.DataFrame([
        (t, 'tok:S', 'mm:1', 'collateral', 1e6, nan, nan), (t, 'mm:1', 'tok:C', 'borrow', 3e5, nan, nan),   # C lent against S
        (t, 'tok:C', 'mm:2', 'collateral', 1e6, nan, nan), (t, 'mm:2', 'tok:USDC', 'borrow', 1e5, nan, nan),
        (t, 'tok:PT-C', 'mm:3', 'collateral', 1e6, nan, nan), (t, 'mm:3', 'tok:USDC', 'borrow', 1e5, nan, nan),
        (t, 'mv:1', 'mm:1', 'allocation', 4e5, nan, nan), (t, 'mv:1', 'mm:2', 'allocation', 1e5, nan, nan),
        (t, 'mv:1', 'mm:9', 'allocation', 5e5, nan, nan),
        (t, 'mv:2', 'mm:3', 'allocation', 3e5, nan, nan), (t, 'mv:2', 'mm:1', 'allocation', 5e3, nan, nan),
        # Aave-style pools: A holds S (collateral) and C (collateral through e-mode only); B holds W (S's wrapper)
        # and PT-C as collateral, and C supplied without collateral use; C is borrowed from both
        (t, 'tok:S', 'pool:A', 'pool_supply', 2e6, 0.8, 0.0), (t, 'tok:C', 'pool:A', 'pool_supply', 5e5, 0.0, 0.9),
        (t, 'tok:W', 'pool:B', 'pool_supply', 1e5, 0.75, 0.0), (t, 'tok:PT-C', 'pool:B', 'pool_supply', 7e5, 0.7, nan),
        (t, 'tok:C', 'pool:B', 'pool_supply', 9e5, 0.0, 0.0),
        (t, 'pool:A', 'tok:C', 'pool_borrow', 4e5, nan, nan), (t, 'pool:B', 'tok:C', 'pool_borrow', 2e5, nan, nan),
        (t, 'pool:B', 'tok:S', 'pool_borrow', 6e4, nan, nan)],
        columns=['t', 'src', 'dst', 'etype', 'usd', 'liq_threshold', 'emode_ltv'])
    static = pd.DataFrame([('tok:PT-C', 'tok:C', 'derivative'), ('tok:W', 'tok:S', 'wrapper')], columns=['src', 'dst', 'etype'])
    g = mc.GraphLinks(edges, static, ['C', 'S', 'W'])
    f = g.features('S', 'C', t + 5 * H)
    assert abs(f['log_lend_cand_vs_seed'] - np.log1p(3e5)) < 1e-9 and f['log_lend_seed_vs_cand'] == 0
    # vault 1: S-family 4e5, C-family 1e5 (of 1e6 total); vault 2: S 5e3, C-family (via PT-C) 3e5 (of 3.05e5)
    assert abs(f['log_vault_overlap'] - np.log1p(1e5 + 5e3)) < 1e-9
    share = (1e5 * 4e5 / 1e6 + 3e5 * 5e3 / 3.05e5) / (1e5 + 3e5)
    assert abs(f['vault_share'] - share) < 1e-12 and f['shared_vaults'] == 1 and f['same_family'] == 0
    # pools: collateral overlap min(2e6, 5e5) in A + min(1e5, 7e5) in B (C's plain supply in B does not count);
    # C borrowed against S-family collateral: min(2e6, 4e5) in A + min(1e5, 2e5) in B
    assert abs(f['log_pool_overlap'] - np.log1p(5e5 + 1e5)) < 1e-9
    assert abs(f['log_pool_lend_cand_vs_seed'] - np.log1p(4e5 + 1e5)) < 1e-9
    r = g.features('C', 'S', t + 5 * H)                                           # the other way round
    assert abs(r['log_pool_overlap'] - f['log_pool_overlap']) < 1e-12
    assert abs(r['log_pool_lend_cand_vs_seed'] - np.log1p(min(7e5, 6e4))) < 1e-9 and r['log_lend_seed_vs_cand'] == f['log_lend_cand_vs_seed']
    w = g.features('S', 'W', t + 5 * H)
    assert w['same_family'] == 1 and all(w[c] == 0 for c in mc.GRAPH)            # family pairs: FAM covers them
    assert g.features('S', 'C', t - H)['log_vault_overlap'] == 0                 # before the first snapshot: nothing
    # an edge file without the Aave columns: no supply counts as collateral, the Morpho features are unchanged
    old = mc.GraphLinks(edges.drop(columns=['liq_threshold', 'emode_ltv']), static, ['C', 'S', 'W']).features('S', 'C', t + 5 * H)
    assert old['log_pool_overlap'] == 0 and old['log_pool_lend_cand_vs_seed'] == 0
    assert all(old[c] == f[c] for c in mc.GRAPH_MORPHO)


def test_follow_features_use_closed_weeks_only():
    syms = ['S', 'C', 'Z']
    ep = pd.DataFrame({'symbol': ['S', 'C', 'S', 'C', 'Z'],
                       't0': [0, 10 * H, 1000 * H, 1050 * H, 2000 * H]})
    pairs = pd.DataFrame({'seed': ['S', 'S'], 'cand': ['C', 'C'], 'tau': [1000 * H + 100 * H, 1000 * H + 200 * H]})
    f = mc.follow_features(pairs, ep, syms)
    a, b = mc.PRIOR
    # at the first tau only S's first episode has a closed week (C followed it); at the second, both
    assert abs(f['codepeg_share'].iat[0] - (1 + a) / (1 + a + b)) < 1e-12
    assert abs(f['codepeg_share'].iat[1] - (2 + a) / (2 + a + b)) < 1e-12
    # follow_rate of C: other assets' closed episodes are S@0 (followed) then S@1000 (followed)
    assert abs(f['follow_rate'].iat[0] - (1 + a) / (1 + a + b)) < 1e-12 and abs(f['follow_rate'].iat[1] - (2 + a) / (2 + a + b)) < 1e-12


def test_contagion_pairs_have_no_look_ahead():
    cut = T0 + 500 * D
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a, groups = build(d1, synthetic())
        b, _ = build(d2, synthetic(cut=cut))
        cols = groups['OWN'] + groups['SIM'] + groups['FAM'] + groups['GRAPH']
        pa = a[a['tau'] <= cut].set_index(['seed_id', 'cand'])[cols].sort_index()
        pb = b[b['tau'] <= cut].set_index(['seed_id', 'cand'])[cols].sort_index()
        assert pa.index.equals(pb.index) and len(pa) > 200
        same = np.isclose(pa.to_numpy(float), pb.to_numpy(float), rtol=1e-5, atol=1e-6, equal_nan=True)
        bad = [c for c, ok in zip(cols, same.all(axis=0)) if not ok]
        assert not bad, bad
        assert len(a[a['tau'] > cut]) != len(b[b['tau'] > cut]) or not a.loc[a['tau'] > cut, 'z_now'].reset_index(drop=True).equals(
            b.loc[b['tau'] > cut, 'z_now'].reset_index(drop=True))


def test_seed_metrics_by_hand():
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], float)
    s = -np.arange(12, dtype=float)                                       # ranks the rows in order
    m = rc.seed_metrics(y, s)
    assert abs(m['ap'] - (1 / 2 + 2 / 5) / 2) < 1e-12 and m['p5'] == 0.4 and m['hit5'] == 1.0
    idcg = 1 + 1 / np.log2(3)
    assert abs(m['ndcg10'] - (1 / np.log2(3) + 1 / np.log2(6)) / idcg) < 1e-12


def test_planted_vault_channel_is_found():
    with tempfile.TemporaryDirectory() as d:
        table, groups = build(d, synthetic())
        lk = table[groups['GRAPH']].gt(0).any(axis=1)
        assert lk.any() and table.loc[lk, 'y168'].mean() > 3 * table.loc[~lk & (table['same_family'] == 0), 'y168'].mean()
        sys.argv = ['x', '--data', f'{d}/model/contagion_severe.csv.gz', '--out', f'{d}/model', '--bootstrap', '200',
                    '--inference-bootstrap', '99', '--models', 'random', 'hist', 'follow', 'logit_own', 'logit_sim', 'logit_fam', 'logit_graph']
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main()
        res = {r['window']: r for r in json.loads(Path(f'{d}/model/contagion_severe_results.json').read_text())}
        assert set(res) == {72, 168}
        r = res[168]
        for p in ('time', 'loco'):
            mm = r['protocols'][p]['models']
            assert mm['logit_graph']['pooled_ap'] > mm['logit_fam']['pooled_ap'], (p, mm['logit_graph'], mm['logit_fam'])
            assert mm['logit_graph']['ap'] > mm['random']['ap'] + 0.1
            assert 0 <= mm['random']['p5'] <= 1 and mm['random']['p5_ci'][0] <= mm['random']['p5'] <= mm['random']['p5_ci'][1]
        gr = [c for c in r['inference']['coef'] if c['block'] == 'GRAPH']     # the vault features are collinear here: one stays
        assert len(gr) == 1 and gr[0]['odds_ratio'] > 1 and gr[0]['p'] < 0.05 and r['inference']['wald_graph']['p_boot'] < 0.05
        assert r['links']['difference_ci'][0] > 0
        md_text = Path(f'{d}/model/contagion_severe_results.md').read_text()
        assert '## Followers within 168 h' in md_text and 'Collapses in the test period' in md_text
        p = pd.read_csv(f'{d}/model/contagion_severe_preds.csv.gz')
        assert set(p['protocol']) == {'time', 'loco'}
        assert p.groupby(['window', 'protocol', 'model']).size().groupby(level=[0, 1]).nunique().eq(1).all()
        # the trees and the 'all' seed set run too
        sys.argv = ['x', '--data', f'{d}/model/contagion_severe.csv.gz', '--out', f'{d}/model', '--bootstrap', '50',
                    '--inference-bootstrap', '0', '--windows', '72', '--protocols', 'time', '--models', 'random', 'hgb_fam', 'hgb_graph']
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main()
        r72 = next(r for r in json.loads(Path(f'{d}/model/contagion_severe_results.json').read_text()) if r['window'] == 72)
        assert set(r72['protocols']) == {'time', 'loco'}                              # the partial run kept the rest
        assert {'logit_graph', 'hgb_graph'} <= set(r72['protocols']['time']['models']) and r72['inference']['bootstrap_draws'] > 0
        p = pd.read_csv(f'{d}/model/contagion_severe_preds.csv.gz')
        assert {'logit_graph', 'hgb_graph'} <= set(p.loc[(p['window'] == 72) & (p['protocol'] == 'time'), 'model'])
        allp, _ = build(d, synthetic(), seeds='all')
        assert allp['seed_id'].nunique() > table['seed_id'].nunique()



def test_no_channel_no_graph_gain():
    """Same vault groups, but followers do not depend on them: the graph must not look useful."""
    with tempfile.TemporaryDirectory() as d:
        build(d, synthetic(seed=1, p_link=0.06, p_other=0.06))
        sys.argv = ['x', '--data', f'{d}/model/contagion_severe.csv.gz', '--out', f'{d}/model', '--bootstrap', '300',
                    '--inference-bootstrap', '199', '--windows', '168', '--protocols', 'time', 'loco',
                    '--models', 'random', 'logit_fam', 'logit_graph']
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main()
        r = json.loads(Path(f'{d}/model/contagion_severe_results.json').read_text())[0]
        for p in ('time', 'loco'):
            assert r['protocols'][p]['pairs']['logit_graph - logit_fam']['ap_ci'][0] <= 0, r['protocols'][p]['pairs']
        assert r['inference']['wald_graph']['p_boot'] > 0.05, r['inference']['wald_graph']


def test_semi_synthetic_power_rises_with_the_planted_risk():
    import power_contagion as pc
    with tempfile.TemporaryDirectory() as d:
        build(d, synthetic(p_link=0.04))                     # no channel in the data: the power check plants it
        args = type('A', (), {'data': f'{d}/model/contagion_severe.csv.gz', 'out': f'{d}/model', 'window': 168,
                              'rr': [1.0, 12.0], 'reps': 8, 'bootstrap': 200, 'seed': 0})
        with contextlib.redirect_stdout(io.StringIO()):
            rows = {r['rr']: r for r in pc.run(args)}
        assert rows[12.0]['mean_linked_followers'] > 3 * rows[1.0]['mean_linked_followers']
        assert rows[12.0]['power_rates'] >= 0.75 and rows[1.0]['power_rates'] <= 0.25
        assert rows[12.0]['mean_dap'] > rows[1.0]['mean_dap']
        assert (Path(d) / 'model' / 'contagion_power.md').exists()

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
