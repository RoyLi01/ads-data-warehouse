from __future__ import annotations

"""
ODS landing job: CSV -> Parquet (dt partitioned) + Hive registration.

Outputs:
- `warehouse/ods/ad_event_log` (partitionBy dt)
- `warehouse/ods/conversion_log` (partitionBy dt)
- `warehouse/ods/ad_cost` (partitionBy dt)

Notes:
- Fact tables are registered into the `ods` Hive database after parquet landing.
- ODS dimension/config tables are handled by `jobs/ingest_ods_dims.py`.
"""

import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from common.spark_session import build_spark


def _project_root() -> Path:
    """
    Resolve repo root directory.

    Assumption: this file lives at `<repo>/jobs/ingest_ods.py`.
    """
    return Path(__file__).resolve().parents[1]


def _warehouse_table_path(warehouse_dir: Path, layer: str, table: str) -> Path:
    """
    Standardize physical table paths under `warehouse/`.

    Example:
    - layer="ods", table="ad_event_log" -> warehouse/ods/ad_event_log
    """
    return warehouse_dir / layer / table


def _read_csv(spark, path: Path, schema: StructType) -> DataFrame:
    """
    Read a header CSV with an explicit schema.

    Why explicit schema:
    - avoids CSV inference drift between runs
    - keeps downstream Parquet schemas stable (important for incremental dt reads)
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing ODS CSV: {path}")
    return spark.read.option("header", "true").schema(schema).csv(str(path))


def _sql_string(s: str) -> str:
    return s.replace("'", "''")


def _ensure_ods_fact_table(
    spark,
    table_name: str,
    columns_sql: str,
    location: Path,
) -> None:
    spark.sql("CREATE DATABASE IF NOT EXISTS ods")
    spark.sql(
        f"""
CREATE TABLE IF NOT EXISTS {table_name} (
  {columns_sql}
)
USING PARQUET
PARTITIONED BY (dt STRING)
LOCATION '{_sql_string(str(location))}'
""".strip()
    )


def _repair_partitioned_table(spark, table_name: str) -> None:
    spark.sql(f"MSCK REPAIR TABLE {table_name}")
    print(f"[HIVE] Repaired partitions for {table_name}")


def main() -> None:
    # Entry point of this job:
    # Run as: `python jobs/ingest_ods.py`
    # Parses CLI args so you can switch input/output dirs without editing code
    parser = argparse.ArgumentParser(description="Ingest ODS CSVs (data/ods/*.csv) into Parquet partitioned by dt.")
    parser.add_argument(
        "--input_dir",
        default=str(_project_root() / "data" / "ods"),
        help="ODS CSV directory (contains *.csv)",
    )
    parser.add_argument(
        "--warehouse_dir",
        default=str(_project_root() / "warehouse"),
        help="Warehouse base dir",
    )
    args = parser.parse_args()

    # Convert to Path for safer/clearer path joins (e.g. input_dir / "ods_ad_event_log.csv").
    input_dir = Path(args.input_dir)
    warehouse_dir = Path(args.warehouse_dir)

    # Create SparkSession using the project's shared settings.
    spark = build_spark(app_name="00_ingest_ods")

    # ------------------- ods_ad_event_log -------------------
    # Event log:
    # explicit schema to prevent CSV inference drift
    # event_time cast to timestamp so later DWD/DWS can do time/window operations safely
    event_schema = StructType(
        [
            StructField("event_id", StringType(), True),
            StructField("event_time", StringType(), True),
            StructField("dt", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("ad_id", StringType(), True),
            StructField("campaign_id", StringType(), True),
            StructField("creative_id", StringType(), True),
            StructField("site_id", StringType(), True),
            StructField("event_type", StringType(), True),
            StructField("device", StringType(), True),
            # Enriched fields from `data_generator/generate_ods.py`
            # Keep them here so ODS Parquet preserves the raw/enriched payload for traceability.
            StructField("session_id", StringType(), True),
            StructField("ad_slot_id", StringType(), True),
            StructField("page_id", StringType(), True),
            StructField("is_valid", IntegerType(), True),
            StructField("ip", StringType(), True),
            StructField("user_agent", StringType(), True),
        ]
    )
    event_csv = input_dir / "ods_ad_event_log.csv"
    ad_event_log = _read_csv(spark, event_csv, event_schema).withColumn(
        "event_time", F.to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss")
    )
    # Repartition by dt to reduce small files per partition in local runs.
    (
        ad_event_log.repartition(F.col("dt"))
        .write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(_warehouse_table_path(warehouse_dir, "ods", "ad_event_log")))
    )
    _ensure_ods_fact_table(
        spark,
        "ods.ad_event_log",
        """
event_id STRING,
event_time TIMESTAMP,
user_id STRING,
ad_id STRING,
campaign_id STRING,
creative_id STRING,
site_id STRING,
event_type STRING,
device STRING,
session_id STRING,
ad_slot_id STRING,
page_id STRING,
is_valid INT,
ip STRING,
user_agent STRING
""".strip(),
        _warehouse_table_path(warehouse_dir, "ods", "ad_event_log"),
    )
    _repair_partitioned_table(spark, "ods.ad_event_log")

    # ------------------- ods_conversion_log -------------------
    # Conversion log:
    # - conv_time cast to timestamp
    # - gmv_amount cast to double (CSV is string by default)
    conv_schema = StructType(
        [
            StructField("conv_id", StringType(), True),
            StructField("conv_time", StringType(), True),
            StructField("dt", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("ad_id", StringType(), True),
            StructField("campaign_id", StringType(), True),
            StructField("order_id", StringType(), True),
            StructField("gmv_amount", StringType(), True),
        ]
    )
    conv_csv = input_dir / "ods_conversion_log.csv"
    conversion_log = (
        _read_csv(spark, conv_csv, conv_schema)
        .withColumn("conv_time", F.to_timestamp("conv_time", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("gmv_amount", F.col("gmv_amount").cast(DoubleType()))
    )
    (
        conversion_log.repartition(F.col("dt"))
        .write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(_warehouse_table_path(warehouse_dir, "ods", "conversion_log")))
    )
    _ensure_ods_fact_table(
        spark,
        "ods.conversion_log",
        """
conv_id STRING,
conv_time TIMESTAMP,
user_id STRING,
ad_id STRING,
campaign_id STRING,
order_id STRING,
gmv_amount DOUBLE
""".strip(),
        _warehouse_table_path(warehouse_dir, "ods", "conversion_log"),
    )
    _repair_partitioned_table(spark, "ods.conversion_log")

    # ------------------- ods_ad_cost -------------------
    # Cost log: daily campaign cost at (dt, campaign_id) grain.
    # Cast cost to double for later KPI math (CPC/CPM/ROI etc.)
    cost_schema = StructType(
        [
            StructField("dt", StringType(), True),
            StructField("campaign_id", StringType(), True),
            StructField("cost", StringType(), True),
        ]
    )
    cost_csv = input_dir / "ods_ad_cost.csv"
    ad_cost = _read_csv(spark, cost_csv, cost_schema).withColumn("cost", F.col("cost").cast(DoubleType()))
    (
        ad_cost.repartition(F.col("dt"))
        .write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(_warehouse_table_path(warehouse_dir, "ods", "ad_cost")))
    )
    _ensure_ods_fact_table(
        spark,
        "ods.ad_cost",
        """
campaign_id STRING,
cost DOUBLE
""".strip(),
        _warehouse_table_path(warehouse_dir, "ods", "ad_cost"),
    )
    _repair_partitioned_table(spark, "ods.ad_cost")

    spark.stop()


if __name__ == "__main__":
    main()

