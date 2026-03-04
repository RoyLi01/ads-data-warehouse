#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Ensure local module imports work for spark-submit drivers.
export PYTHONPATH="$ROOT_DIR"

# Avoid DNS issues in some environments.
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
export SPARK_LOCAL_HOSTNAME="${SPARK_LOCAL_HOSTNAME:-localhost}"

spark_submit() {
  local script="$1"

  # Prefer pyspark-bundled spark-submit if available (pip-installed pyspark).
  local pyspark_submit
  pyspark_submit="$(python - <<'PY'
import pathlib
import pyspark
print(pathlib.Path(pyspark.__file__).resolve().parent / "bin" / "spark-submit")
PY
)"

  local cmd="spark-submit"
  if [[ -x "$pyspark_submit" ]]; then
    cmd="$pyspark_submit"
  fi

  "$cmd" \
    --conf "spark.driver.extraPythonPath=$PYTHONPATH" \
    --conf "spark.executor.extraPythonPath=$PYTHONPATH" \
    "$script"
}

python data_generator/generate_data.py
spark_submit spark/jobs/00_ingest_ods.py
spark_submit spark/jobs/01_build_dwd.py
spark_submit spark/jobs/02_build_dws.py
spark_submit spark/jobs/03_build_ads.py
spark_submit spark/jobs/04_data_quality_check.py

echo "All done."

