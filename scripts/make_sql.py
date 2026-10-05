#!/usr/bin/env python3
"""Render the BigQuery SQL templates in sql/ with the registry's token list.

Usage:
  python scripts/fetch_rates_rpc.py --decimals          # once: writes config/decimals.json
  python scripts/make_sql.py                            # tables go to dataset 'depeg' of the project you run them in
  python scripts/make_sql.py --project my-gcp-project --dataset depeg     # or name the project explicitly
Then, in the BigQuery console (or with bq query --use_legacy_sql=false < file), in a dataset in the US
multi-region; paste the files from sql/generated/, not the templates in sql/:
  sql/generated/02_dex_trades_2023.sql ... _2026.sql   one query per year; each reads that year of the
                                                       public token_transfers table and stores only the
                                                       trades of the registry tokens (a small table)
  sql/generated/03_dex_prices_hourly.sql               hourly DEX prices -> save as CSV to
                                                       data/dex_prices_hourly.csv
  sql/generated/01_hourly_blocks.sql                   optional: hour -> first block (for
                                                       fetch_rates_rpc.py --blocks)
The editor shows how many bytes a query will scan before it runs; step 02 reads seven columns of the
public token_transfers table (Google's goog_blockchain_ethereum_mainnet_us dataset), step 03 only the
small yearly tables.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_registry  # noqa: E402

QUOTES = {'USDC': ('USD', 6), 'USDT': ('USD', 6), 'DAI': ('USD', 18), 'USDS': ('USD', 18), 'WETH': ('ETH', 18)}


def decimals_table():
    out = {}
    meta = ROOT / 'data' / 'token_meta_llama.csv'
    if meta.exists():
        with open(meta) as fh:
            for r in csv.DictReader(fh):
                if r.get('decimals'):
                    out[r['symbol']] = int(float(r['decimals']))
    dj = ROOT / 'config' / 'decimals.json'
    if dj.exists():  # on-chain decimals() wins over DefiLlama metadata
        out.update({k: v for k, v in json.loads(dj.read_text()).items() if v is not None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='', help='GCP project of your dataset (default: the project the query runs in)')
    ap.add_argument('--dataset', default='depeg')
    ap.add_argument('--source-dataset', default='bigquery-public-data.goog_blockchain_ethereum_mainnet_us')
    ap.add_argument('--start', default='2023-01-01')
    ap.add_argument('--end-excl', default='2026-10-01')
    ap.add_argument('--min-usd', type=float, default=100.0, help='drop trades smaller than this (USD quotes)')
    ap.add_argument('--min-eth', type=float, default=0.05, help='drop trades smaller than this (WETH quotes)')
    ap.add_argument('--floor-usd', type=float, default=1.0, help='step 02 keeps trades from this size (USD quotes)')
    ap.add_argument('--floor-eth', type=float, default=0.0005, help='step 02 keeps trades from this size (WETH quotes)')
    args = ap.parse_args()

    _, assets = load_registry()
    dec = decimals_table()
    rows, missing = [], []
    for a in assets:
        if a['symbol'] in QUOTES:
            peg, d = QUOTES[a['symbol']]
            rows.append((a['address'].lower(), a['symbol'], 'quote', dec.get(a['symbol'], d), peg))
        elif a['in_scope']:
            if a['symbol'] not in dec:
                missing.append(a['symbol'])
                continue
            rows.append((a['address'].lower(), a['symbol'], 'asset', dec[a['symbol']], a['peg']))
    if missing:
        sys.exit(f"decimals unknown for {missing}: run scripts/fetch_rates_rpc.py --decimals (or fetch_llama_prices.py) first")

    token_list = ',\n'.join(f"    '{r[0]}'" for r in rows)
    structs = ',\n'.join(f"    STRUCT('{r[0]}' AS address, '{r[1]}' AS symbol, '{r[2]}' AS role, {r[3]} AS decimals, '{r[4]}' AS peg)"
                         for r in rows)
    prefix = f'{args.project}.{args.dataset}' if args.project else args.dataset
    base = dict(prefix=prefix, source_dataset=args.source_dataset,
                start=args.start, end_excl=args.end_excl, n_tokens=len(rows), token_list=token_list,
                token_structs=structs, min_usd=args.min_usd, min_eth=args.min_eth,
                floor_usd=args.floor_usd, floor_eth=args.floor_eth)
    if args.floor_usd > args.min_usd or args.floor_eth > args.min_eth:
        sys.exit('--floor-usd/--floor-eth (step 02) must not exceed --min-usd/--min-eth (step 03)')
    out = ROOT / 'sql' / 'generated'
    out.mkdir(parents=True, exist_ok=True)
    tmpl = {p.name: p.read_text() for p in (ROOT / 'sql').glob('*.sql.tmpl')}
    (out / '01_hourly_blocks.sql').write_text(tmpl['01_hourly_blocks.sql.tmpl'].format(**base))
    y0, y1 = int(args.start[:4]), int(args.end_excl[:4])
    for y in range(y0, y1 + 1):
        s, e = max(args.start, f'{y}-01-01'), min(args.end_excl, f'{y + 1}-01-01')
        if s < e:
            (out / f'02_dex_trades_{y}.sql').write_text(
                tmpl['02_dex_trades.sql.tmpl'].format(**{**base, 'start': s, 'end_excl': e, 'year': y}))
    (out / '03_dex_prices_hourly.sql').write_text(tmpl['03_dex_prices_hourly.sql.tmpl'].format(**base))
    print(f'{len(rows)} tokens; wrote {sorted(p.name for p in out.glob("*.sql"))} to {out}')


if __name__ == '__main__':
    main()
