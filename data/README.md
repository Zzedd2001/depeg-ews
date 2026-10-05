# depeg-ews data archive (v1.1)

The files below are everything the paper's tables and figures are computed from. They are derived from public sources by the scripts in this repository; times are UTC, and `ts` columns are Unix seconds. Unzip the archive in the repository root so that the files land in `data/`.

The derived data are released under CC BY 4.0. The upstream sources keep their own terms: the DefiLlama coins API, the Morpho API, Ethereum state read from an archive node, and Google's public Ethereum dataset on BigQuery (`bigquery-public-data.goog_blockchain_ethereum_mainnet_us`).

## Inputs

| File | Made by | Contents |
| --- | --- | --- |
| `prices_llama_hourly.csv.gz` | `fetch_llama_prices.py` | Hourly prices: `symbol, llama_key, hour, ts, price, confidence, source` (DefiLlama, within 30 minutes of each hour, confidence ≥ 0.9) |
| `token_meta_llama.csv` | `fetch_llama_prices.py` | DefiLlama keys, fallback keys, decimals and coverage per asset |
| `rates.csv.gz` | `fetch_rates_rpc.py` | On-chain exchange rates every 6 h: `symbol, time, ts, block, rate, ok` |
| `dex_prices_hourly.csv` | `sql/03_dex_prices_hourly.sql.tmpl` on BigQuery | DEX trades per asset, quote token and hour: trade count, parties, volume, VWAP and price quartiles `p25, p50, p75` |
| `morpho/markets.csv`, `morpho/vaults.csv` | `fetch_morpho.py` | 401 Morpho markets (tokens, LLTV, oracle and its class) and 418 MetaMorpho vaults |
| `morpho/feed_descriptions.json` | `fetch_morpho.py` | `description()` of every oracle price feed, read on chain |
| `morpho/market_history_{day,hour}.csv.gz` | `fetch_morpho.py` | Market states in long format: `market_id, ts, field, value` (supply, borrow and collateral in USD and tokens) |
| `morpho/vault_allocation_{day,hour}.csv.gz` | `fetch_morpho.py` | Vault allocations per market in long format |
| `lending/aave_reserves.csv`, `lending/aave_emode.csv`, `lending/aave_reserves_meta.json` | `fetch_aave_reserves.py` | Every reserve of the Aave v3 Core, Lido and EtherFi markets and SparkLend every 6 h: supply, debt, oracle price, LTV, liquidation threshold and bonus, caps, flags; e-mode categories |

## Exposure graph

| File | Contents |
| --- | --- |
| `graph/nodes.csv`, `graph/edges_static.csv` | Tokens, markets, vaults and pools; wrapper and derivative relations |
| `graph/edges.csv.gz` | Daily snapshots of collateral, supply, borrow, vault-allocation and pool edges with USD values, LLTV/LTV, oracle class and freeze flags |
| `graph/token_features.csv.gz` | The 21 exposure measures per asset and daily snapshot |
| `graph_hourly/token_features.csv.gz` | The same measures every hour (used by the models) |
| `graph/market_flags.csv`, `graph/build_report.md` | Markets handled by the cleaning rules; build log |

## Labels and the DEX check

| File | Contents |
| --- | --- |
| `labels/labels_hourly.csv.gz` | Deviation, episode flag and 24/72-h onset labels per asset-hour: `hour, symbol, dev, in_episode, y24, y72, obs`; `obs` is 1 at hours with a price sample of their own and 0 otherwise (a deviation carried forward over a gap of up to 3 h, which features read and episodes do not, or none) |
| `labels/episodes.csv` | 456 detected episodes (444 after the quality filters; the `suspect` column marks the 12 others) with start, end, the sample that completed the start condition (`confirmed_at`; the episode is known from the later of it and one hour after the start), deepest and held depth, severity |
| `labels/coverage.csv` | Price coverage, reference type and data-quality flags per asset |
| `labels/checkpoint_report.md`, `labels/sensitivity.csv`, `labels/known_events_check.csv` | Summary report, episode counts under other thresholds, and the 15 known events |
| `labels/source_check.{md,csv}`, `labels/source_check_placebo.csv`, `labels/source_check_dex_only.csv` | Every episode and placebo window checked against DEX trades; dips seen only in DEX trades |
| `labels/exclude_contradicted.csv` | The 59 contradicted episodes left out in the robustness run |
| `labels_robust/` | Labels without those 59 episodes |

## Models and results

`model/` holds the main run and `model_robust/` the robustness run; the modeling table `dataset.pkl.gz` is not included and is rebuilt by `make_dataset.py` in about 10 seconds.

| File | Contents |
| --- | --- |
| `results.{md,json}`, `compare_preds.{md,json}` | Onset metrics for every model and horizon; paired AP differences with asset-block bootstrap intervals |
| `preds_<model>_<24h,72h>.csv.gz` | Onset test-period scores: `symbol, ts, y, has_exposure, score` (read with `float_precision='round_trip'` to get exactly the scores the models wrote) |
| `escalation.csv.gz`, `escalation_results.{md,json}`, `escalation_preds_*.csv.gz` | Escalation table (one row per episode and decision hour), results and scores |
| `contagion_{severe,all}.csv.gz`, `contagion_*_results.{md,json}`, `contagion_*_preds.csv.gz` | Seed–candidate pairs, contagion results and scores |
| `contagion_power.{md,json}` | Semi-synthetic power of the contagion link tests |
| `*_features.json` | Feature groups used by each task |
