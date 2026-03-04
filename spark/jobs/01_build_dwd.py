from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark.common.spark_session import build_spark


def _project_root() -> Path:
    # spark/jobs/*.py -> spark/jobs -> spark -> project root
    return Path(__file__).resolve().parents[2]


def _table_path(warehouse_dir: Path, layer: str, table: str) -> Path:
    return warehouse_dir / layer / table


def _read_table(spark, warehouse_dir: Path, layer: str, table: str) -> DataFrame:
    return spark.read.parquet(str(_table_path(warehouse_dir, layer, table)))


def _write_partitioned(df: DataFrame, warehouse_dir: Path, layer: str, table: str) -> None:
    (
        df.write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(_table_path(warehouse_dir, layer, table)))
    )


def _standardize_ts(col: str) -> F.Column:
    """
    Standardize time field to Spark TimestampType.

    Input may already be timestamp (from ODS landing) or string in 'yyyy-MM-dd HH:mm:ss'.
    """
    return F.to_timestamp(F.col(col).cast("string"), "yyyy-MM-dd HH:mm:ss")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DWD detail tables from ODS parquet.")
    parser.add_argument("--warehouse_dir", default=str(_project_root() / "warehouse"))
    parser.add_argument(
        "--dt",
        default=None,
        help="Optional partition date filter (YYYY-MM-DD). If omitted, process all dt partitions found.",
    )
    args = parser.parse_args()

    warehouse_dir = Path(args.warehouse_dir)
    dt_filter = args.dt

    spark = build_spark(app_name="01_build_dwd")

    # ===================== Input tables (ODS) =====================
    # - ods/ad_event_log: grain = 1 row per event (impression or click)
    # - ods/conversion_log: grain = 1 row per conversion
    ad_event_log = _read_table(spark, warehouse_dir, "ods", "ad_event_log")
    conversion_log = _read_table(spark, warehouse_dir, "ods", "conversion_log")

    if dt_filter:
        ad_event_log = ad_event_log.where(F.col("dt") == dt_filter)
        conversion_log = conversion_log.where(F.col("dt") == dt_filter)

    # ===================== DWD: impression detail =====================
    # Table: dwd_ad_impression_detail
    # Grain: 1 row per impression event (event_type='impression')
    # Cleaning rules:
    # - Filter event_type = 'impression'
    # - De-duplicate by event_id (and dt for safety)
    # - Standardize event_time to timestamp
    dwd_ad_impression_detail = (
        ad_event_log.where(F.col("event_type") == F.lit("impression"))
        .withColumn("event_time", _standardize_ts("event_time"))
        .dropDuplicates(["dt", "event_id"])
        .select(
            "dt",
            "event_id",
            "event_time",
            "user_id",
            "ad_id",
            "campaign_id",
            "creative_id",
            "site_id",
            "device",
        )
    )
    _write_partitioned(dwd_ad_impression_detail, warehouse_dir, "dwd", "dwd_ad_impression_detail")

    # ===================== DWD: click detail =====================
    # Table: dwd_ad_click_detail
    # Grain: 1 row per click event (event_type='click')
    # Cleaning rules:
    # - Filter event_type = 'click'
    # - De-duplicate by event_id (and dt for safety)
    # - Standardize event_time to timestamp
    dwd_ad_click_detail = (
        ad_event_log.where(F.col("event_type") == F.lit("click"))
        .withColumn("event_time", _standardize_ts("event_time"))
        .dropDuplicates(["dt", "event_id"])
        .select(
            "dt",
            "event_id",
            "event_time",
            "user_id",
            "ad_id",
            "campaign_id",
            "creative_id",
            "site_id",
            "device",
        )
    )
    _write_partitioned(dwd_ad_click_detail, warehouse_dir, "dwd", "dwd_ad_click_detail")

    # ===================== DWD: conversion detail =====================
    # Table: dwd_ad_conversion_detail
    # Grain: 1 row per conversion (conv_id)
    # Cleaning rules:
    # - De-duplicate by conv_id (and dt for safety)
    # - Standardize conv_time to timestamp
    # - Ensure gmv_amount is double
    dwd_ad_conversion_detail = (
        conversion_log.withColumn("conv_time", _standardize_ts("conv_time"))
        .withColumn("gmv_amount", F.col("gmv_amount").cast("double"))
        .dropDuplicates(["dt", "conv_id"])
        .select(
            "dt",
            "conv_id",
            "conv_time",
            "user_id",
            "ad_id",
            "campaign_id",
            "order_id",
            "gmv_amount",
        )
    )
    _write_partitioned(dwd_ad_conversion_detail, warehouse_dir, "dwd", "dwd_ad_conversion_detail")

    spark.stop()


if __name__ == "__main__":
    main()

