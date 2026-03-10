#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Ensure local module imports work.
export PYTHONPATH="$ROOT_DIR"

# Avoid DNS issues in some environments.
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
export SPARK_LOCAL_HOSTNAME="${SPARK_LOCAL_HOSTNAME:-localhost}"

# Speed knobs:
# - FAST=1: skip expensive DQ/count() prints and skip dq_check
# - DT/START_DT/DAYS: override default dates
FAST="${FAST:-0}"
DT="${DT:-2026-03-01}"
START_DT="${START_DT:-$DT}"
DAYS="${DAYS:-3}"

python data_generator/generate_ods.py --start_dt "$START_DT" --days "$DAYS"

# Initialize Hive databases / external tables first.
python jobs/init_hive.py

# ODS landing (CSV -> Parquet partitioned tables)
python jobs/ingest_ods.py
python jobs/ingest_ods_dims.py

# Official warehouse pipeline (incremental by dt)
if [[ "$FAST" == "1" ]]; then
  python jobs/build_dwd.py --dt "$DT" --no_dq
  python jobs/build_dws.py --dt "$DT" --no_dq
  python jobs/build_ads.py --dt "$DT" --no_dq
else
  python jobs/build_dwd.py --dt "$DT"
  python jobs/build_dws.py --dt "$DT"
  python jobs/build_ads.py --dt "$DT"
fi

# Data quality checks on the official pipeline outputs
if [[ "$FAST" != "1" ]]; then
  python jobs/dq_check.py --dt "$DT"
fi

echo "All done."

