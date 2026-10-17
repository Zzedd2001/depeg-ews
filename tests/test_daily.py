"""Offline tests for daily_baseline.py and economic_magnitude.py on synthetic data.

python tests/test_daily.py        (needs scipy for the tail model)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import daily_baseline as db  # noqa: E402
import economic_magnitude as em  # noqa: E402

DAY = 86400
D0 = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp())
ND = 500
SYMS = ['USDe', 'sUSDe', 'wstETH']
META = {'USDe': {'peg': 'USD'}, 'sUSDe': {'peg': 'USD'}, 'wstETH': {'peg': 'ETH'}}


def closes(seed=0, perturb_after=None):
    """Daily closes in threshold units, with dips, and a daily ETH price."""
    rng = np.random.default_rng(seed)
    idx = pd.Index(D0 + DAY * np.arange(ND), name='ts')
    z = pd.DataFrame({s: rng.normal(0, 0.1, ND) for s in SYMS}, index=idx)
    for s in SYMS:
        for d in rng.choice(np.arange(10, ND - 5), size=12, replace=False):
            z.loc[idx[d]:idx[d + 2], s] -= rng.uniform(0.6, 2.0)
    z.iloc[40:43, 1] = np.nan                                  # a gap
    eth = pd.Series(3000 * np.exp(np.cumsum(rng.normal(0, 0.02, ND))), index=idx)
    if perturb_after is not None:                              # rewrite everything after the cut
        late = idx > perturb_after
        z.loc[late] = z.loc[late] * -3 + 1.0
        eth.loc[late] *= 0.5
    return z, eth


def test_daily_features_have_no_look_ahead():
    cut = D0 + 300 * DAY
    za, ea = closes()
    zb, eb = closes(perturb_after=cut)
    fa = db.daily_features(za, ea, {'sUSDe': 'USDe'}, META, SYMS)
    fb = db.daily_features(zb, eb, {'sUSDe': 'USDe'}, META, SYMS)
    assert set(fa) == set(fb) and len(fa) > 20
    for name in fa:
        a, b = fa[name].loc[:cut], fb[name].loc[:cut]
        same = (a == b) | (a.isna() & b.isna())
        assert same.all().all(), name                          # nothing at or before the cut moved
    assert not fa['dz_now'].loc[cut + DAY:].equals(fb['dz_now'].loc[cut + DAY:])   # the perturbation is real
    # the wrapper sees its underlying, and a close beyond the threshold counts once per day
    assert np.allclose(fa['d_und_now']['sUSDe'].dropna(), za['USDe'][fa['d_und_now']['sUSDe'].notna()])
    beyond = (za['USDe'] <= -1).astype(float)
    assert np.allclose(fa['d_beyond_30d']['USDe'], beyond.rolling(30, min_periods=1).sum())


def test_tail_scores_have_no_look_ahead():
    """AR and tail parameters come from training days only; the variance recursion runs forward in time."""
    train_end = D0 + 250 * DAY
    cut = D0 + 350 * DAY
    za, _ = closes()
    zb, _ = closes(perturb_after=cut)
    pegs = {s: META[s]['peg'] for s in SYMS}
    pa, ia = db.fit_tail(za, pegs, train_end)
    pb, ib = db.fit_tail(zb, pegs, train_end)
    assert ia['gpd_shape'] == ib['gpd_shape'] and ia['pooled_ar'] == ib['pooled_ar']
    a, b = pa.loc[:cut], pb.loc[:cut]
    assert ((a == b) | (a.isna() & b.isna())).all().all()
    assert not pa.loc[cut + DAY:].equals(pb.loc[cut + DAY:])
    assert float(np.nanmin(pa.to_numpy())) >= 0 and float(np.nanmax(pa.to_numpy())) <= 1


def test_tail_probability_rises_near_the_threshold():
    za, _ = closes()
    p, _ = db.fit_tail(za, {s: META[s]['peg'] for s in SYMS}, D0 + 250 * DAY)
    near = p['USDe'][za['USDe'] < -0.8].mean()
    calm = p['USDe'][za['USDe'].abs() < 0.1].mean()
    assert near > calm


def test_random_alert_probability_and_runs():
    # k alerts out of n hours hit at least one of w hours: compare with simulation
    rng = np.random.default_rng(0)
    n, k, w = 400, 20, 24
    sim = np.mean([np.isin(np.arange(w), rng.choice(n, k, replace=False)).any() for _ in range(20000)])
    assert abs(em.p_random(w, k, n) - sim) < 0.01
    assert em.p_random(0, k, n) == 0.0 and em.p_random(n, k, n) == 1.0
    # runs: consecutive alerted hours of one asset; a run counts as followed by a start if any hour is positive
    H = 3600
    sym = np.array(['A', 'A', 'A', 'A', 'B', 'B'])
    ts = np.array([0, H, 3 * H, 4 * H, 0, H])
    y = np.array([0, 0, 0, 1, 0, 0])
    assert em.runs(sym, ts, y) == (3, 1)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
