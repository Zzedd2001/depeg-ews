#!/usr/bin/env python3
"""Figures for the paper, drawn from the result files (no model is refitted).

  fig2_dex_check         DEX check of the labelled episodes against placebo windows (Section 3)
  fig3_exposure_effect   change in AP from adding lending exposure, every task and protocol, 95 % intervals;
                         filled = main labels, hollow = without the episodes DEX trades contradict (Section 6)
  fig4_warning_horizon   (a) onset: share of test episodes with an alert at least k hours before the start
                         (5 % alert rate, 72-h models); (b) escalation: hours from the decision hour until the
                         depeg holds -5 % (escalating episodes)
  fig5_power             semi-synthetic power of the contagion link tests (power_contagion.py)
Numbers follow the paper's order of first citation (Fig. 1 is the schematic drawn in the draft).

Usage:
  python scripts/make_figures.py                 # writes figures/*.pdf and figures/*.png
Needs matplotlib. Inputs: data/model/, data/model_robust/, data/labels/ (see each function).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, epoch_seconds, label_columns, not_suspect, observed_only  # noqa: E402

H = 3600
INK, GREY, LIGHT = '#1b1b1b', '#8c8c8c', '#d9d9d9'
BLUE, ORANGE, GREEN, RED = '#1f5a96', '#d9822b', '#3a8a4f', '#b8443a'


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'savefig.dpi': 300,
                         'text.hinting': 'no_hinting'})   # PNG text rasterized the same whatever the local matplotlibrc
    return plt


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None})   # same bytes on a rerun
    fig.savefig(out / f'{name}.png', bbox_inches='tight', dpi=300)
    print(f'wrote {out / name}.pdf and .png')


# ---------------------------------------------------------------- figure 2
def exposure_rows(model_dir):
    """(group, label, diff, lo, hi) for one result directory; missing results are skipped."""
    d = Path(model_dir)
    rows = []
    cp = json.loads((d / 'compare_preds.json').read_text()) if (d / 'compare_preds.json').exists() else []
    for h in (24, 72):
        r = next((x for x in cp if x['horizon'] == h and x['a'] == 'hgb_PE' and x['b'] == 'hgb_P'), None)
        for sub, name in (('exposed', 'hours with exposure'), ('unexposed', 'hours without exposure')):
            if r:
                v = r['subsets'][sub]
                rows.append(('Onset (trees)', f'{h} h, {name}', v['diff'], *v['ci']))
    esc = json.loads((d / 'escalation_results.json').read_text()) if (d / 'escalation_results.json').exists() else []
    e = next((x for x in esc if x['landmark'] == 1 and x['outcome'] == 'severe'), None)
    if e:
        for pair, model in (('logit_coreE - logit_core', 'Regression'), ('hgb_PE - hgb_P', 'Trees')):
            for prot, pname in (('time', 'time split'), ('loao', 'asset held out')):
                v = e['protocols'].get(prot, {}).get('pairs', {}).get(pair)
                if v:
                    rows.append(('Escalation (1 h in)', f'{model}, {pname}', v['ap'], *v['ap_ci']))
    for fname, seeds in (('contagion_severe_results.json', 'severe seeds'), ('contagion_all_results.json', 'all seeds')):
        c = json.loads((d / fname).read_text()) if (d / fname).exists() else []
        w = next((x for x in c if x['window'] == 168), None)
        if not w:
            continue
        prots = (('time', 'time split'), ('loco', 'cluster held out')) if seeds == 'severe seeds' else (('time', 'time split'),)
        for prot, pname in prots:
            v = w['protocols'].get(prot, {}).get('pairs', {}).get('logit_graph - logit_fam')
            if v:
                rows.append(('Contagion (7 days)', f'Regression, {seeds}, {pname}', v['ap'], *v['ap_ci']))
    return rows


def exposure_effect(plt, root, out):
    main = exposure_rows(root / 'data' / 'model')
    robust = {(g, l): (x, lo, hi) for g, l, x, lo, hi in exposure_rows(root / 'data' / 'model_robust')}
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    y, ticks, labels, group_y = 0, [], [], []
    last = None
    for g, label, x, lo, hi in main:
        if g != last:
            y -= 0.6
            group_y.append((y, g))
            y -= 1.0
            last = g
        ax.plot([lo, hi], [y + 0.14, y + 0.14], color=BLUE, lw=1.4, solid_capstyle='butt')
        ax.plot(x, y + 0.14, 'o', color=BLUE, ms=4.2, zorder=3)
        if (g, label) in robust:
            rx, rlo, rhi = robust[(g, label)]
            ax.plot([rlo, rhi], [y - 0.16, y - 0.16], color=ORANGE, lw=1.1, solid_capstyle='butt')
            ax.plot(rx, y - 0.16, 'o', mfc='white', mec=ORANGE, mew=1.1, ms=4.0, zorder=3)
        ticks.append(y)
        labels.append(label)
        y -= 1.0
    ax.axvline(0, color=GREY, lw=0.8, ls='--', zorder=0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for gy, g in group_y:
        ax.text(-0.01, gy, g, transform=ax.get_yaxis_transform(), ha='right', va='center', fontweight='bold', fontsize=8)
    ax.set_ylim(y + 0.4, 0.2)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Change in average precision from adding lending exposure (95% interval)')
    from matplotlib.lines import Line2D
    n_all, n_rob = (len(not_suspect(pd.read_csv(root / 'data' / d / 'episodes.csv'))) for d in ('labels', 'labels_robust'))
    handles = [Line2D([], [], color=BLUE, marker='o', ms=4.2, lw=1.4, label=f'All {n_all:,} episodes'),
               Line2D([], [], color=ORANGE, marker='o', mfc='white', ms=4.0, lw=1.1,
                      label=f'Without the {n_all - n_rob:,} episodes DEX trades contradict')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis='x', color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)
    save(fig, out, 'fig3_exposure_effect')
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def onset_horizon(model_dir, episodes, model, h=72, budget=0.05, kmax=72):
    p = pd.read_csv(Path(model_dir) / f'preds_{model}_{h}h.csv.gz', float_precision='round_trip')   # the scores exactly
    # exactly the top `budget` of test asset-hours, ties broken by time as in run_baselines (a quantile threshold
    # would flag more hours for models with many tied scores, such as the per-asset base rate)
    top = np.argsort(-p['score'].to_numpy(), kind='stable')[:int(round(budget * len(p)))]
    alerts = p.iloc[np.sort(top)]
    lo, hi = p['ts'].min(), p['ts'].max()
    ep = episodes[(episodes['ts'] > lo) & (episodes['ts'] <= hi + h * H)]
    by_sym = {s: g['ts'].to_numpy() for s, g in alerts.groupby('symbol')}
    lead = []
    for s, t0 in zip(ep['symbol'], ep['ts']):
        a = by_sym.get(s, np.array([]))
        a = a[(a >= t0 - h * H) & (a <= t0 - H)]
        lead.append((t0 - a.min()) / H if len(a) else 0.0)
    lead = np.array(lead)
    k = np.arange(1, kmax + 1)
    return k, np.array([(lead >= x).mean() for x in k]), len(ep)


def escalation_waits(root):
    path = root / 'data' / 'labels' / 'labels_hourly.csv.gz'
    lab = pd.read_csv(path, usecols=label_columns(path))
    lab['ts'] = epoch_seconds(lab['hour'])
    lab['dev'] = observed_only(lab)                          # -5 % must hold over two price samples
    dev = {s: g.set_index('ts')['dev'].sort_index() for s, g in lab.groupby('symbol')}
    esc = pd.read_csv(root / 'data' / 'model' / 'escalation.csv.gz',
                      usecols=['symbol', 't0', 'ts', 'landmark', 'y_severe', 'split'])
    pos = esc[(esc['landmark'] == 1) & (esc['y_severe'] == 1)]
    waits, split = [], []
    for s, t0, tau, sp in zip(pos['symbol'], pos['t0'], pos['ts'], pos['split']):
        w = dev[s].loc[t0:].dropna()
        v, idx = w.to_numpy(), w.index.to_numpy()
        hit = next((idx[i] for i in range(1, len(v)) if max(v[i - 1], v[i]) <= -0.05), None)
        if hit is not None:
            waits.append((hit - tau) / H)                       # from the decision hour (start + 1 h, or later)
            split.append(sp)
    return np.array(waits), np.array(split)


def warning_horizon(plt, root, out):
    ep = not_suspect(pd.read_csv(root / 'data' / 'labels' / 'episodes.csv'))
    ep['ts'] = epoch_seconds(ep['start'])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'wspace': 0.32})
    for model, name, color, ls in (('hgb_P', 'Trees (price features)', BLUE, '-'), ('logit_P', 'Logistic (price features)', GREEN, '--'),
                                   ('rule_dev', 'Deviation rule', ORANGE, '-.'), ('base_rate', 'Per-asset base rate', GREY, ':')):
        k, share, n = onset_horizon(root / 'data' / 'model', ep, model)
        a1.plot(k, 100 * share, color=color, ls=ls, lw=1.3, label=name)
        print(f'  {model}: alerted at least 1/12/24/48/72 h ahead: '
              + ', '.join(f'{100 * share[x - 1]:.1f}%' for x in (1, 12, 24, 48, 72)))
    a1.set_xlim(1, 72)
    a1.set_xticks([1, 12, 24, 36, 48, 60, 72])
    a1.set_ylim(0, 100)
    a1.set_xlabel('Hours before the depeg starts')
    a1.set_ylabel('Episodes alerted at least this early (%)')
    a1.set_title(f'(a) Onset, 72-h models, {n} test episodes', loc='left')
    a1.legend(frameon=False, loc='upper right')
    a1.grid(color=LIGHT, lw=0.5)
    waits, split = escalation_waits(root)
    for sel, name, color, ls in ((np.ones(len(waits), bool), f'All escalations ({len(waits)})', BLUE, '-'),
                                 (split == 'test', f'Test period ({int((split == "test").sum())})', ORANGE, '--')):
        w = np.sort(waits[sel])
        a2.step(np.r_[0, w], np.r_[0, np.arange(1, len(w) + 1) / len(w)] * 100, where='post', color=color, ls=ls, lw=1.3, label=name)
    a2.set_xscale('symlog', linthresh=24)
    a2.set_xlim(0, 2500)
    a2.set_xticks([0, 6, 12, 24, 72, 168, 720, 2160])
    a2.set_xticklabels(['0', '6', '12', '24', '72', '168', '720', '2160'])
    a2.set_ylim(0, 100)
    a2.set_xlabel('Hours after the decision until −5% holds')
    a2.set_ylabel('Escalating episodes (%)')
    a2.set_title('(b) Escalation: time left to act', loc='left')
    med = float(np.median(waits))
    print(f'  escalation waits: n={len(waits)}, median {med:.1f} h, within 6 h {100 * (waits <= 6).mean():.0f}%, '
          f'within 24 h {100 * (waits <= 24).mean():.0f}%')
    a2.axvline(med, color=GREY, lw=0.8, ls=':')
    a2.text(med * 1.06, 6, f'median {med:.0f} h', color=GREY, fontsize=7)
    a2.legend(frameon=False, loc='upper left')
    a2.grid(color=LIGHT, lw=0.5)
    save(fig, out, 'fig4_warning_horizon')
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def dex_check(plt, root, out):
    lab = root / 'data' / 'labels'
    c = pd.read_csv(lab / 'source_check.csv')
    c = c[c['suspect'].fillna('') == '']
    p = pd.read_csv(lab / 'source_check_placebo.csv')
    statuses = ['confirmed', 'partial', 'contradicted', 'unverified']
    colors = [GREEN, '#9cc9a5', RED, LIGHT]
    names = ['Confirmed', 'Partly confirmed', 'Contradicted', 'Cannot be judged']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9), sharey=True, gridspec_kw={'wspace': 0.08})
    axes[0].invert_yaxis()                                   # shared y axis: invert once
    for ax, (title, filt) in zip(axes, (('(a) All windows', lambda d: d),
                                        ('(b) Windows with an hour of two or more trades', lambda d: d[d['dex_robust_hours'] > 0]))):
        sizes = []
        for i, (name, d) in enumerate((('Labelled episodes', filt(c)), ('Placebo windows', filt(p)))):
            sizes.append(len(d))
            share = d['status'].value_counts(normalize=True).reindex(statuses, fill_value=0) * 100
            left = 0.0
            for st, col in zip(statuses, colors):
                ax.barh(i, share[st], left=left, color=col, height=0.6, edgecolor='white', lw=0.5)
                if share[st] >= 6:
                    ax.text(left + share[st] / 2, i, f'{share[st]:.0f}', ha='center', va='center', fontsize=7,
                            color='white' if st in ('confirmed', 'contradicted') else INK)
                left += share[st]
        ax.set_xlim(0, 100)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Labelled episodes', 'Placebo windows'])
        ax.set_title(f'{title} ({sizes[0]:,} and {sizes[1]:,})', loc='left')
        ax.set_xlabel('Share of windows (%)')
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=col, label=n) for col, n in zip(colors, names)], loc='lower center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, out, 'fig2_dex_check')
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def power(plt, root, out):
    path = root / 'data' / 'model' / 'contagion_power.json'
    if not path.exists():
        print('skip fig. 5: run scripts/power_contagion.py first')
        return
    r = pd.DataFrame(json.loads(path.read_text())).sort_values('rr')
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    ax.plot(r['rr'], 100 * r['power_rates'], 'o-', color=BLUE, lw=1.3, ms=3.5, label='Linked pairs follow more often')
    ax.plot(r['rr'], 100 * r['power_ranking'], 's--', color=ORANGE, lw=1.3, ms=3.2, label='Links raise ranking AP')
    if 'share_at_or_below_observed' in r:
        ax.plot(r['rr'], 100 * r['share_at_or_below_observed'], '^:', color=GREY, lw=1.2, ms=3.5,
                label='As low as the observed rates')
    ax.set_xscale('log', base=2)
    ax.set_xticks(r['rr'])
    ax.set_xticklabels([f'{x:g}' for x in r['rr']])
    ax.set_ylim(-3, 103)
    ax.set_xlabel('Relative risk of following for linked pairs')
    ax.set_ylabel('Share of replicates (%)')
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0, 1.0), fontsize=7)      # above the plot: no line crosses it
    ax.grid(color=LIGHT, lw=0.5)
    save(fig, out, 'fig5_power')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--out', default=str(ROOT / 'figures'))
    ap.add_argument('--only', nargs='+', choices=['2', '3', '4', '5'], default=['2', '3', '4', '5'])
    args = ap.parse_args()
    plt = setup()
    root, out = Path(args.root), Path(args.out)
    for k in args.only:
        {'2': dex_check, '3': exposure_effect, '4': warning_horizon, '5': power}[k](plt, root, out)


if __name__ == '__main__':
    main()
