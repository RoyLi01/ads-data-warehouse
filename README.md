# Offline Ad Data Warehouse with Hive Metastore Project

This is a local offline advertising analytics warehouse built with `Hive Metastore + Spark SQL + PySpark + Parquet`.

This project models a realistic ad-data pipeline from raw event ingestion to reporting outputs through a classic `ODS -> DWD -> DWS -> ADS` architecture. It includes fact and dimension ingestion, SQL-first warehouse jobs, a user-tagging module, and a reproducible benchmark for data skew and salting-based optimization.

## Highlights
- Four-layer warehouse architecture: `ODS -> DWD -> DWS -> ADS`
- Hive Metastore used as the metadata/catalog layer, with Parquet as the physical storage layer
- SQL-first DWD / DWS / ADS jobs, with Python kept as a lightweight orchestrator
- Explicit fact and dimension modeling for event, conversion, cost, user, ad, and ad-slot data
- User tag snapshot pipeline based on daily and rolling 7-day features
- ADS outputs for KPI overview, campaign ranking, advertiser dashboard, and tag effectiveness
- Reproducible 20M-row benchmark for campaign-level data skew
- Salting + two-stage aggregation experiment to mitigate hot-key skew on `campaign_id`

## Tech Stack
- Python
- PySpark
- Spark SQL
- Hive Metastore (embedded Derby for local development)
- Parquet
- Local Spark (`local[*]`)
- Benchmark / skew experiment tooling under `benchmark/`

## Project Architecture

```text
Raw CSV generator
  ↓
ODS
  ↓
DWD
  ↓
DWS
  ↓
ADS
```

### ODS
- Ingests raw fact CSVs into partitioned Parquet under `warehouse/ods/...`
- Registers ODS facts and dimensions into the `ods` Hive database
- Stores:
  - event log
  - conversion log
  - campaign cost
  - user profile
  - ad metadata
  - ad slot metadata

### DWD
- Cleans and standardizes raw ODS inputs
- Deduplicates facts, filters invalid traffic, and normalizes event / conversion timestamps
- Enriches facts with business dimensions such as advertiser, ad type, landing type, product, and slot metadata
- Produces dimension snapshots and detail facts:
  - user dimension
  - ad dimension
  - ad-slot dimension
  - impression detail
  - click detail
  - conversion detail
  - campaign-day cost

### DWS
- Builds daily subject-area aggregates on top of DWD
- Standardizes campaign, advertiser, and user metrics such as `impressions`, `clicks`, `conversions`, `gmv`, `cost`, `ctr`, `cvr`, `rpm`, and `roi`
- Adds a user-tagging module:
  - static tag dictionary
  - daily user tag snapshot
  - daily tag quality table

### ADS
- Produces reporting-friendly outputs from DWS
- Includes:
  - KPI overview
  - campaign ranking
  - advertiser dashboard
  - tag effectiveness report

## Main Tables

### ODS
| Table | Grain | Purpose |
|---|---|---|
| `ods.ad_event_log` | `dt + event_id` | Raw impression / click event stream landed to Parquet and Hive |
| `ods.conversion_log` | `dt + conv_id` | Conversion facts with order and GMV information |
| `ods.ad_cost` | `dt + campaign_id` | Daily campaign cost input for downstream ROI/cost analysis |
| `ods.user_profile` | `user_id` | User dimension source with demographics and registration date |
| `ods.ad_meta` | `ad_id + campaign_id` | Ad metadata source with advertiser, ad type, landing type, and product |
| `ods.ad_slot` | `ad_slot_id` | Ad slot / placement metadata |

### DWD
| Table | Grain | Purpose |
|---|---|---|
| `dwd.dim_user` | `dt + user_id` | Daily user dimension snapshot |
| `dwd.dim_ad` | `dt + ad_id + campaign_id` | Daily ad dimension snapshot aligned to fact enrichment grain |
| `dwd.dim_ad_slot` | `dt + ad_slot_id` | Daily ad-slot dimension snapshot |
| `dwd.ad_impression_detail` | `dt + event_id` | Cleaned and enriched impression facts |
| `dwd.ad_click_detail` | `dt + event_id` | Cleaned and enriched click facts |
| `dwd.ad_conversion_detail` | `dt + conv_id` | Cleaned and enriched conversion facts |
| `dwd.campaign_day_cost` | `dt + campaign_id` | Standardized campaign-day cost table for DWS cost aggregation |

### DWS
| Table | Grain | Purpose |
|---|---|---|
| `dws.campaign_daily` | `dt + campaign_id` | Daily campaign metrics for performance analysis |
| `dws.advertiser_daily` | `dt + advertiser_id` | Daily advertiser aggregates for dashboarding |
| `dws.user_daily` | `dt + user_id` | Daily user behavior aggregate |
| `dws.user_tag_snapshot` | `dt + user_id` | User tag snapshot built from same-day and 7-day rolling features |
| `dws.tag_quality_daily` | `dt + tag_name` | Daily tag coverage and effectiveness metrics |
| `dws.tag_dict` | `tag_id` | Built-in tag dictionary and rule definitions |

### ADS
| Table | Grain | Purpose |
|---|---|---|
| `ads.kpi_overview_daily` | `dt` | Executive KPI overview of traffic, conversion, GMV, cost, and ROI |
| `ads.campaign_ranking_daily` | `dt + campaign_id` | Campaign leaderboard by GMV and ROI |
| `ads.advertiser_dashboard_daily` | `dt + advertiser_id` | Advertiser dashboard with growth and retention indicators |
| `ads.tag_effectiveness_daily` | `dt + tag_name` | Tag-level effectiveness summary |

## Key Modules

### Fact and Dimension Modeling
- `jobs/ingest_ods.py` lands ODS fact tables to partitioned Parquet and registers them in Hive.
- `jobs/ingest_ods_dims.py` ingests user / ad / ad-slot dimensions into the ODS layer.
- `jobs/build_dwd.py` is SQL-dominant and converts ODS data into cleaned, enriched warehouse facts and dimension snapshots.

### User Tagging Module
- Implemented inside `jobs/build_dws.py`
- Uses both daily features and rolling 7-day features
- Produces:
  - `dws.tag_dict`
  - `dws.user_tag_snapshot`
  - `dws.tag_quality_daily`

### Reporting Layer
- `jobs/build_ads.py` generates dashboard-ready outputs from DWS
- Covers overview KPIs, campaign ranking, advertiser reporting, and tag effectiveness analysis

## Data Skew Benchmark

The project includes a standalone benchmark under `benchmark/` to reproduce and explain data skew on `campaign_id` aggregation.

### Experiment Goal
Validate how a hot `campaign_id` affects Spark aggregation runtime, partition balance, and resource pressure under a campaign-level aggregation workload similar to DWS.

### Experiment Setup
- Dataset size: `20,000,000` rows per distribution
- Hot-key setup: one `campaign_id` accounts for roughly `80%` of rows in the skewed dataset
- Stress benchmark mode:
  - fixed shuffle partitions
  - repartition by `campaign_id`
  - sort within partitions
  - campaign-level aggregation

### Distribution Impact: `uniform` vs `skewed`

Observed locally with `20M` rows and `shuffle_partitions=4`:

| Scenario | Elapsed Time | Partition Skew Ratio |
|---|---:|---:|
| `uniform` | `36.31s` | `1.14` |
| `skewed` | `62.67s` | `3.43` |

- `slowdown_ratio = 1.73`
- The skewed run also triggered memory-pressure warnings during execution

### Skew Mitigation: `skewed` vs `skewed_salted`

The benchmark also includes a salted version of the skewed aggregation:

- Input is still the same `skewed` dataset
- The hot `campaign_id` is split into multiple salted sub-keys
- Aggregation is executed in two stages:
  1. `dt + campaign_id + salt`
  2. `dt + campaign_id`

Observed locally with `20M` rows, `shuffle_partitions=4`, and `salt_buckets=8`:

| Scenario | Elapsed Time | Partition Skew Ratio |
|---|---:|---:|
| `skewed` | `32.03s` | `3.43` |
| `skewed_salted` | `23.95s` | `1.80` |

- `improvement_ratio = 1.34`
- The salted run materially reduced partition imbalance and improved runtime

### Benchmark Conclusion
This benchmark shows two important points:

1. A hot `campaign_id` can create strong partition imbalance, long-tail tasks, and higher resource pressure during aggregation.
2. Salting + two-stage aggregation is an effective mitigation strategy for hot-key skew, even in a local Spark environment.

## How to Run

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate ODS Data
For a realistic DWS tagging run, generate at least 7 days:

```bash
python data_generator/generate_ods.py --start_dt 2026-03-01 --days 7
```

### 3. Initialize Hive Metadata
```bash
PYTHONPATH=. python jobs/init_hive.py
```

### 4. Ingest ODS Facts and Dimensions
```bash
PYTHONPATH=. python jobs/ingest_ods.py
PYTHONPATH=. python jobs/ingest_ods_dims.py
```

### 5. Build DWD and DWS
If you want correct 7-day user-tag features, run DWD and DWS sequentially by day:

```bash
for dt in 2026-03-01 2026-03-02 2026-03-03 2026-03-04 2026-03-05 2026-03-06 2026-03-07; do
  PYTHONPATH=. python -m jobs.build_dwd --dt "$dt" --no_dq
  PYTHONPATH=. python -m jobs.build_dws --dt "$dt" --no_dq
done
```

### 6. Build ADS
```bash
PYTHONPATH=. python -m jobs.build_ads --dt 2026-03-07 --no_dq
```

### 7. Optional Smoke Run
For a quick end-to-end smoke run:

```bash
bash run_all.sh
```

Note: `run_all.sh` is useful for a single-date smoke test, while the 7-day sequential DWD/DWS run above is recommended when demonstrating rolling user-tag features.

## Benchmark Usage

### Generate Benchmark Data
```bash
python benchmark/generate_skew_data.py --rows 20000000
```

### Run `uniform` / `skewed` / `skewed_salted`
```bash
PYTHONPATH=. python benchmark/run_campaign_skew_benchmark.py --modes all --benchmark_mode stress --shuffle_partitions 4 --salt_buckets 8
```

### Compare Only `skewed` vs `skewed_salted`
```bash
PYTHONPATH=. python benchmark/run_campaign_skew_benchmark.py --modes skewed skewed_salted --benchmark_mode stress --shuffle_partitions 4 --salt_buckets 8
```

### Benchmark Outputs
- Data: `benchmark/data/...`
- Aggregation outputs: `benchmark/results/<mode>/campaign_daily_stress`
- Result files:
  - `benchmark/results/benchmark_results.csv`
  - `benchmark/results/benchmark_results.jsonl`

## Repository Structure

```text
.
├── benchmark/
│   ├── generate_skew_data.py
│   ├── run_campaign_skew_benchmark.py
│   ├── data/
│   ├── results/
│   └── tmp/
├── common/
│   └── spark_session.py
├── data/
│   └── ods/
├── data_generator/
│   └── generate_ods.py
├── docs/
├── jobs/
│   ├── init_hive.py
│   ├── ingest_ods.py
│   ├── ingest_ods_dims.py
│   ├── build_dwd.py
│   ├── build_dws.py
│   ├── build_ads.py
│   └── dq_check.py
├── scripts/
├── warehouse/
│   ├── ods/
│   ├── dwd/
│   ├── dws/
│   ├── ads/
│   ├── hive_warehouse/
│   └── metastore_db/
├── requirements.txt
├── run_all.sh
└── README.md
```

## Future Extensions
- Move from local embedded Hive metastore to a distributed Spark / Hive environment
- Add richer data-quality checks and reconciliation workflows
- Extend the warehouse with more business domains, attribution logic, and campaign lifecycle metrics
- Add more benchmark modes, such as skewed joins or skew-aware window computations
- Add orchestration and scheduling once the project is migrated beyond local development

