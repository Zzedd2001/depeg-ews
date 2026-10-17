#!/usr/bin/env python3
"""Exclusion list for the trade-supported robustness run: every episode that DEX trades do not confirm.

Reads data/labels/source_check.csv (compare_sources.py) and writes data/labels/exclude_not_trade_supported.csv
with the episodes that passed the quality filters but are neither confirmed nor partly confirmed by trades
(contradicted, or not judged for lack of trades). Relabelling with this list keeps only trade-supported episodes:

  python scripts/trade_supported.py
  python scripts/make_labels.py --prices data/prices_llama_hourly.csv.gz data/dex_prices_hourly.csv --primary defillama \\
      --rates data/rates.csv.gz --exclude-episodes data/labels/exclude_not_trade_supported.csv --out-dir data/labels_traded
  (then the dataset, escalation, contagion and baseline steps with --labels/--episodes from data/labels_traded and
   outputs to data/model_traded, as in the README)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', default=str(ROOT / 'data' / 'labels' / 'source_check.csv'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'labels' / 'exclude_not_trade_supported.csv'))
    args = ap.parse_args()
    sc = pd.read_csv(args.check)
    sc = sc[sc['suspect'].fillna('').astype(str).str.strip() == '']
    keep = sc['status'].isin(['confirmed', 'partial'])
    sc.loc[~keep, ['symbol', 'start']].to_csv(args.out, index=False)
    print(sc['status'].value_counts().to_string())
    print(f'kept {int(keep.sum())} trade-supported episodes; {int((~keep).sum())} written to {args.out}')


if __name__ == '__main__':
    main()
