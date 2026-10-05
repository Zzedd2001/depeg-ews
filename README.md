# depeg-ews

An early-warning benchmark for depegs of synthetic and yield-bearing stablecoins and liquid-staking tokens on Ethereum, with the code that rebuilds every number in the paper.

> **Onset, Escalation, and Contagion: What Warns of Stablecoin and Liquid-Staking Depegs?**
> Zhengdong Zhu. Submitted to *IEEE Access*, 2026.
> Data archive: Zenodo, DOI to be added on publication of the record.

Version 1.1 (October 2026). The benchmark follows 52 Ethereum assets hour by hour from January 2023 to September 2026 and asks three questions along the life of a depeg.

| Task | Decision at | Positive when | Size |
| --- | --- | --- | --- |
| Onset | every asset-hour | an episode starts within 24 h (also 72 h) | 987,399 asset-hours; 346,006 in the test year |
| Escalation | 1 h after an episode starts, or later if a price gap delays its confirmation (also 6 h, 24 h) | its held depth reaches −5% | 418 episodes, 61 escalate |
| Contagion | the hour a seed's depeg turns severe | the candidate starts an episode within 7 days (also 72 h) | 73 seeds, 2,705 seed–candidate pairs |

Labels come from fixed price rules, with no manual annotation and no language model anywhere in the pipeline: 444 depeg episodes on 44 assets, checked against trades reconstructed from Ethereum ERC-20 transfers. A lending-exposure graph (401 Morpho markets, 418 MetaMorpho vaults, the Aave v3 Core, Lido and EtherFi markets, and SparkLend) supplies exposure features and links between assets.

Main results on the test year (October 2025 to September 2026):

- **Onset.** Logistic regression on price features reaches an average precision (AP) of 0.078 at 24 h, 7.5 times the share of positive hours (1.04%); gradient-boosted trees on the same features (AP 0.077, scores averaged over five seeds) alert ahead of 67% of the 156 test episodes while flagging 5% of asset-hours.
- **Escalation.** The depth reached in the first hour more than doubles the base rate (AP 0.488 against 0.194); escalating depegs take a median of 21 h after the decision hour to hold −5%.
- **Contagion.** A regression on the candidate's own depeg history and current stress puts a follower in the top five for 32 of the 34 test seeds that had one (AP 0.515 against 0.149 for a random order).
- **Lending exposure** adds nothing at any stage. In the full sample, exposure levels behave as drifting asset fingerprints, and the observed follow rates make a public lending channel that doubles a linked asset's risk unlikely. Without the 59 episodes that DEX trades contradict, exposure still adds nothing.

## Repository layout

| Path | Contents |
| --- | --- |
| `config/assets.json` | 59 Ethereum assets: 52 prediction targets (8 synthetic dollars, 4 RWA-backed dollars, 6 yield-bearing wrappers, 14 crypto-collateralized dollars, 13 liquid-staking and 7 liquid-restaking tokens) and 7 reference assets, with addresses, categories, reference-value rules and exchange-rate contracts |
| `config/label_rules.json` | Thresholds, run lengths, recovery rule, horizons, sensitivity grid and the time split |
| `config/known_events.json` | 15 documented depeg events used to check the labels |
| `config/lending_pools.json`, `config/wrappers.json` | Aave v3 and SparkLend addresses; wrapper relations (sUSDe→USDe, wstETH→stETH, ...) |
| `scripts/fetch_llama_prices.py` | Hourly prices from the free DefiLlama coins API |
| `scripts/fetch_rates_rpc.py` | On-chain exchange rates every 6 h from an archive node (one Multicall3 call per time point) |
| `scripts/make_sql.py`, `sql/*.sql.tmpl` | BigQuery SQL that reconstructs DEX trades from public ERC-20 transfers and aggregates hourly price quartiles |
| `scripts/make_labels.py` | Deviations, episodes, 24/72-h labels, sensitivity grid, known-event check |
| `scripts/compare_sources.py` | Checks every episode against DEX trades (confirmed, partly confirmed, contradicted, cannot be judged), placebo windows, DEX-only dips |
| `scripts/fetch_morpho.py`, `scripts/fetch_aave_reserves.py` | Morpho markets, oracles and vault allocations (free API); Aave v3 and SparkLend reserves (archive node) |
| `scripts/build_graph.py` | Exposure-graph snapshots, token families (`edges_static.csv`) and 21 exposure measures per asset, using only past data |
| `scripts/make_dataset.py`, `scripts/run_baselines.py`, `scripts/compare_preds.py` | Onset: hourly modeling table (39 price and 45 exposure features), 8 models (tree scores averaged over five seeds), paired comparisons with an asset-block bootstrap |
| `scripts/make_escalation.py`, `scripts/run_escalation.py` | Escalation: episode table at 1 and 6 h (24 h with `--landmarks 1 6 24`), 7 models, leave-one-asset-out, clustered regressions with a pairs cluster bootstrap-t |
| `scripts/make_contagion.py`, `scripts/run_contagion.py` | Contagion: seed–candidate pairs, 10 rankers, leave-one-cluster-out, link follow rates |
| `scripts/power_contagion.py` | Semi-synthetic power of the contagion link tests on the real pairs |
| `scripts/make_figures.py` | Figures 2 to 5 of the paper |
| `scripts/paper_checks.py` | Numbers the paper quotes that the scripts above do not print: recall on repeat episodes, warning horizons, alert load, what random alerts at the same rate would catch, escalation waits, drop-one-asset inference, exposure drift, per-seed AP changes, separation of the link coefficients, a common-shock variant of the power check, synthetic channels, single-seed onset trees and a second set of five seeds |
| `scripts/make_release.py` | Builds the public code tree and the data archive |
| `tests/` | 73 offline tests, including leakage tests for every task and planted-effect power tests |
| `figures/` | Figures 2 to 5 as PDF and PNG |
| `data/README.md` | Description of every data file and its source |

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

Tested with Python 3.10 (pandas 2.3, numpy 2.2, scikit-learn 1.7, scipy 1.15, matplotlib 3.10) and Python 3.13 (pandas 3.0, numpy 2.5, scikit-learn 1.9, scipy 1.18). The results in the data archive were computed with the second set, which `requirements-lock.txt` pins (`pip install -r requirements-lock.txt`). With scikit-learn 1.7 or 1.8 in place of 1.9 (the other packages as pinned), regressions and rules give identical results, but those versions place the trees' histogram bins differently: the five-seed onset trees on price features, with and without exposure, then move by less than 0.006 in AP, and the escalation trees gain 0.051 from exposure on the time split instead of 0.024 (Section 8 of the paper).

## Reproduce the paper from the data archive

Download `depeg-ews-data-v1.1.zip` from the Zenodo record and unzip it in the repository root with `unzip -o depeg-ews-data-v1.1.zip`; it fills `data/` with the inputs and every result file (and replaces `data/README.md` with the same text). Then rerun the models:

```bash
python scripts/make_dataset.py                       # hourly modeling table, ~10 s, ~1.7 GB RAM
python scripts/run_baselines.py                      # onset, 8 models x 24/72 h, ~25 min (five seeds per tree model) -> data/model/results.md
python scripts/compare_preds.py                      # paired AP differences, ~5 min      -> data/model/compare_preds.md
python scripts/make_escalation.py --landmarks 1 6 24
python scripts/run_escalation.py                     # ~2 min                             -> data/model/escalation_results.md
python scripts/make_contagion.py
python scripts/run_contagion.py                      # ~1 min                             -> data/model/contagion_severe_results.md
python scripts/make_contagion.py --seeds all         # every episode from 2024-01-07 as a seed
python scripts/run_contagion.py --seeds all          # both protocols, ~5 min            -> data/model/contagion_all_results.md
python scripts/power_contagion.py --rr 1 2 3 4 6 8 16 --reps 1000   # ~2 min per relative risk -> data/model/contagion_power.md
python scripts/make_figures.py                       # figures/fig2..fig5 (.pdf and .png)
python scripts/paper_checks.py horizon               # and the other subcommands, see the script's header
```

Onset prediction files (`preds_*.csv.gz`) store scores at full precision. `make_figures.py` and `paper_checks.py` read them with `float_precision='round_trip'`, which returns exactly the scores the models wrote; pandas' default parser can be off in the last bit.

| Paper element | Source file |
| --- | --- |
| Table 2, Table A1 | `data/labels/episodes.csv`, `data/labels/checkpoint_report.md`, `data/labels/known_events_check.csv`, `data/labels/source_check.md` |
| Table 4, Section 6.1 | `data/model/results.md`, `data/model/compare_preds.md`; `paper_checks.py repeats`, `horizon`, `chance` |
| Table 5, Section 6.2 | `data/model/escalation_results.md`; `paper_checks.py waits` |
| Table 6, Section 6.3 | `data/model/contagion_severe_results.md`, `data/model/contagion_all_results.md`; `paper_checks.py seeds` |
| Section 5, contagion candidates | `make_contagion.py --candidates known --out data/model_known/contagion_severe.csv.gz`, then `run_contagion.py --data data/model_known/contagion_severe.csv.gz --out data/model_known` (~25 min) |
| Section 6.4 | `data/model/results.md`, `compare_preds.md`, `escalation_results.md`, `contagion_severe_results.md`, `contagion_power.md`; `paper_checks.py drift`, `drop`, `shock`, `synthetic` |
| Section 6.5 | the same files under `data/model_robust/`; `paper_checks.py separation`, and `drop` with `--data data/model_robust/escalation.csv.gz` and the flags in the script's header |
| Section 7 | `paper_checks.py monitor`, `chance`, `waits` |
| Sections 5 and 8, single-seed trees and a second seed set | `paper_checks.py treeseeds`, `paper_checks.py seedset` (~20 min); the scikit-learn 1.7 numbers come from rerunning `run_baselines.py`, `compare_preds.py` and `run_escalation.py` after `pip install scikit-learn==1.7.2` |
| Fig. 2 | `data/labels/source_check.csv`, `data/labels/source_check_placebo.csv` |
| Fig. 3 | `compare_preds.json`, `escalation_results.json`, `contagion_*_results.json` in `data/model/` and `data/model_robust/` |
| Fig. 4 | `data/model/preds_*_72h.csv.gz`, `data/model/escalation.csv.gz`, `data/labels/labels_hourly.csv.gz`, `data/labels/episodes.csv` |
| Fig. 5 | `data/model/contagion_power.json` |

## Rebuild everything from public sources

Each fetcher caches its responses and resumes after an interruption.

**1. Prices** (DefiLlama, no account):

```bash
python scripts/fetch_llama_prices.py --start 2023-01-01 --end 2026-09-30    # up to ~3,900 requests
```

**2. Exchange rates** (any archive-capable Ethereum endpoint; keep the URL in an environment variable and out of every file):

```bash
export ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/<your key>
python scripts/fetch_rates_rpc.py --check       # reads every rate at the latest block
python scripts/fetch_rates_rpc.py --decimals    # writes config/decimals.json
python scripts/fetch_rates_rpc.py               # every 6 h, ~5,500 calls
```

**3. DEX trades** (Google BigQuery; the free sandbox is enough):

```bash
python scripts/make_sql.py                      # writes sql/generated/
```

In the BigQuery console, create a dataset `depeg` in the US multi-region, run `sql/generated/02_dex_trades_2023.sql` to `_2026.sql` (the 2023 query scans about 70 GB), then run `sql/generated/03_dex_prices_hourly.sql` and save the result as `data/dex_prices_hourly.csv`. The queries avoid the sandbox limits: no DML, only trades are stored (10 GB cap), and tables are not partitioned (the sandbox deletes partitions after 60 days).

**4. Labels and the DEX check:**

```bash
python scripts/make_labels.py --prices data/prices_llama_hourly.csv.gz data/dex_prices_hourly.csv \
    --primary defillama --rates data/rates.csv.gz
python scripts/compare_sources.py --rates data/rates.csv.gz              # -> data/labels/source_check.md
python scripts/make_labels.py --prices data/prices_llama_hourly.csv.gz data/dex_prices_hourly.csv \
    --primary defillama --rates data/rates.csv.gz \
    --exclude-episodes data/labels/exclude_contradicted.csv --out-dir data/labels_robust
```

**5. Lending-exposure graph:**

```bash
python scripts/fetch_morpho.py --interval DAY               # ~2,500 requests, ~1 h (with ETH_RPC_URL set it also reads oracle feed descriptions)
python scripts/fetch_morpho.py --interval HOUR              # markets that ever exceeded $1 million, ~5,500 requests
python scripts/fetch_aave_reserves.py --check
python scripts/fetch_aave_reserves.py                       # Aave v3 Core, Lido, EtherFi and SparkLend, every 6 h
python scripts/build_graph.py                               # daily snapshots, links and token families
python scripts/build_graph.py --step-hours 1 --no-edges --out data/graph_hourly    # hourly features, ~1 min, ~3.3 GB RAM
```

The Morpho API allows 750 requests a minute; the fetcher waits 0.5 s between requests and stops rather than waiting when the API asks for a pause of more than 15 minutes. `make_contagion.py` stops if `data/graph/edges_static.csv` (the token families) is missing, rather than treating every token as its own family.

**6. Models:** as in the previous section. The robustness run without the 59 episodes that DEX trades contradict uses the same scripts with other paths:

```bash
python scripts/make_dataset.py --labels data/labels_robust/labels_hourly.csv.gz --episodes data/labels_robust/episodes.csv --out data/model_robust/dataset.pkl.gz
python scripts/run_baselines.py --data data/model_robust/dataset.pkl.gz --episodes data/labels_robust/episodes.csv --out data/model_robust
python scripts/compare_preds.py --dir data/model_robust
python scripts/make_escalation.py --labels data/labels_robust/labels_hourly.csv.gz --episodes data/labels_robust/episodes.csv --out data/model_robust/escalation.csv.gz --landmarks 1 6 24
python scripts/run_escalation.py --data data/model_robust/escalation.csv.gz --out data/model_robust
python scripts/make_contagion.py --labels data/labels_robust/labels_hourly.csv.gz --episodes data/labels_robust/episodes.csv --out data/model_robust/contagion_severe.csv.gz
python scripts/run_contagion.py --data data/model_robust/contagion_severe.csv.gz --out data/model_robust
python scripts/make_contagion.py --labels data/labels_robust/labels_hourly.csv.gz --episodes data/labels_robust/episodes.csv --seeds all --out data/model_robust/contagion_all.csv.gz
python scripts/run_contagion.py --seeds all --data data/model_robust/contagion_all.csv.gz --out data/model_robust --protocols time
```

## Design choices that matter

- **Labels.** The deviation is the market value over a reference value minus one: one dollar for dollar stablecoins, the on-chain exchange rate for wrappers and staking tokens, one ether for stETH, eETH and frxETH, and the trailing 168-h median for five assets without a reliable rate. With a threshold θ of 1% for dollar assets and 2% for ether assets, an episode starts at two consecutive price samples at or below −θ, or at one sample at or below −5%; it closes once 24 consecutive samples lie above −θ/2 and ends at its last sample at or below −θ/2. Hours without a sample neither extend nor break a run; features, but not labels, carry the last deviation forward over gaps of up to 3 h (`obs` in `labels_hourly.csv.gz` is 1 at hours with a sample of their own). Severity is the deepest level held for two consecutive samples.
- **No look-ahead.** Every feature uses only information available at the decision hour. An episode becomes known at the sample that completes its start condition (`confirmed_at` in `episodes.csv`), and no earlier than one hour after its start (`common.known_ts`; 17 episodes become known two to four hours in); onset features, escalation decisions and contagion seeds use that hour. Morpho points are used from one hour after their timestamp, Aave and Spark readings from the block they were read at. `tests/` rewrites all inputs after a cutoff (deviations, every exposure measure, graph links, ether prices, later episodes) and checks that no earlier feature changes, for each task. Rows and candidates are selected with the episodes as dated by `make_labels.py`: onset rows inside episodes are dropped, and contagion candidates inside an episode, or within 24 h of its end, are left out, so an asset whose own episode starts at the decision hour is not a candidate (`make_contagion.py --candidates known` keeps it, as a non-follower).
- **Inference with few clusters.** Intervals resample assets or seed clusters; regression p-values come from a pairs cluster bootstrap-t, because asymptotic cluster-robust tests over-reject when few clusters carry a regressor. Regressions winsorize features at training percentiles, except that a feature whose two percentiles coincide keeps its full range, so a rare nonzero value is not erased.

## Known limitations

- **Prices.** Labels use hourly DefiLlama prices, which blend exchange and on-chain prices and miss intra-hour extremes: ezETH traded down to about $700 within the hour on 24 April 2024, while the hourly series shows −14.3%. The data cover Ethereum mainnet from January 2023, so earlier events such as UST and stETH in 2022 are outside them, and Ethereum prices may not reflect events of assets that trade mainly on other chains (YU, USDX).
- **DEX check.** Trades are reconstructed by pairing ERC-20 transfers, so swaps against native ETH, such as those in the Curve stETH/ETH pool, are invisible, and lending operations and liquidations can pass for trades at the loan-to-value ratio or the liquidation discount. The check therefore uses hourly price quartiles and never confirms a depeg from an hour with a single trade; removing such flows entirely would need prices from pool `Swap` events and a registry of pools.
- **Exposure graph.** It covers Morpho, the Aave v3 Core, Lido and EtherFi markets, and SparkLend on Ethereum. Morpho Vault V2 (which allocates through adapters), Euler, Fluid, Pendle pools and centralized venues are not included, and the depositors behind private vaults are not visible, which is how the documented xUSD → deUSD channel of November 2025 was missed.
- **Pool links.** Aave and Spark are read as reserve totals, which do not say which collateral backs which debt, so the two pool links are upper bounds; exact links would need borrower-level positions. The list of Aave reserves is read at the latest block, so reserves dropped earlier are not read, and e-mode eligibility follows the reserve's own category before Aave v3.2 and the category bitmaps after it.
- **Oracles and vaults.** Oracle classes rely on the `description()` that each price feed reports, and the 57 `custom` oracles are treated as able to see a depeg. Vault allocation histories are as the Morpho API returns them; if it lists only the markets a vault still uses, past allocations to markets it has left are missed.
- **Derivatives and exchange rates.** Pendle and similar derivatives are recognized by symbol (`PT-`, `YT-`, `LP-` or `SY-` followed by a registry symbol); irregular symbols are missed and can be added by address under `by_address` in `config/wrappers.json`. Of the 21 exchange-rate contracts, 19 come from Balancer's reviewed rate-provider registry and are read with `getRate()`; scrvUSD and sdeUSD are read with ERC-4626 `convertToAssets`, and whether sdeUSD follows ERC-4626 in full is unconfirmed.

## Tests

```bash
pip install duckdb sqlglot eth-abi        # optional: SQL logic and ABI encoding tests
python tests/test_make_labels.py
python tests/test_fetchers.py
python tests/test_exposure.py             # Morpho and Aave fetchers against simulated APIs, offline
python tests/test_models.py
python tests/test_escalation.py           # ~1-2 min
python tests/test_contagion.py            # ~1-3 min
python tests/test_release.py
```

## Changes from version 1.0

- Episodes and their held depth are measured on hours with a price sample of their own; in v1.0 an hour carried forward over a short gap counted as a second sample (`make_labels.py --fill-counts` restores that rule).
- An episode closes after 24 consecutive samples back above −θ/2 and ends at its last sample at or below −θ/2, and it counts from the hour it becomes known (the later of its confirming sample, `confirmed_at`, and one hour after its start), not from one hour after its start in every case. Together these rules give 444 episodes instead of 570.
- Winsorizing no longer makes rare features constant: a feature whose training percentiles coincide, such as an indicator or link that is nonzero in fewer than 1% of training rows, keeps its full range.
- `make_contagion.py` stops when the token-family file is missing; Fig. 4 alerts on exactly the top 5% of test asset-hours, ties broken by time as in `run_baselines.py`; escalation waits are counted from the decision hour.
- Onset tree scores are the mean over five random seeds (`run_baselines.py --seeds`): the seed sets the random draw of training rows on which the trees' bins are placed, and a single seed moved the 24-h AP of the price trees between 0.068 and 0.082. A second set of five seeds (5 to 9) moves the averaged price trees' AP by less than 0.001 and keeps the pattern of exposure effects. `requirements-lock.txt` pins the library versions.
- The power check uses 1,000 replicates per relative risk (200 in v1.0); `paper_checks.py` is new; data reports print repository-relative paths.

## Data sources and licenses

The code is released under the MIT License (`LICENSE`). The derived data in the Zenodo archive (labels, features, graph, results) are released under CC BY 4.0. They are built from public sources that keep their own terms: the DefiLlama coins API, the Morpho API, Ethereum state read from an archive node, and Google's public Ethereum dataset on BigQuery (`bigquery-public-data.goog_blockchain_ethereum_mainnet_us`). See `data/README.md`.

## Citation

See `CITATION.cff`. Until the paper is published, please cite it as:

> Z. Zhu, "Onset, escalation, and contagion: What warns of stablecoin and liquid-staking depegs?" submitted to *IEEE Access*, 2026.
