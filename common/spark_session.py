from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pyspark.sql import SparkSession


def _project_root() -> Path:
    # common/spark_session.py -> common -> project root
    return Path(__file__).resolve().parents[1]


def build_spark(app_name: str, extra_conf: Optional[Dict[str, Any]] = None) -> SparkSession:
    """
    Unified SparkSession config for local offline DW jobs.

    Notes:
    - Local mode, Parquet storage
    - AQE enabled (common interview point in China big-data roles)
    - Dynamic partition overwrite to mimic Hive dt=... partitions
    - Hive is used as the metadata/catalog layer, while Parquet remains the storage format
    """
    hive_warehouse_dir = _project_root() / "warehouse" / "hive_warehouse"
    metastore_db_dir = _project_root() / "warehouse" / "metastore_db"

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        # Persist CREATE TABLE / database metadata in the local Hive metastore.
        .config("spark.sql.catalogImplementation", "hive")
        # AQE
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # sane local defaults
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "Asia/Shanghai")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # overwrite only touched partitions (dt)
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        # Keep Hive warehouse and metastore files inside the repo for local runs.
        .config("spark.sql.warehouse.dir", str(hive_warehouse_dir))
        .config("spark.hadoop.hive.metastore.warehouse.dir", str(hive_warehouse_dir))
        .config(
            "spark.hadoop.javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={metastore_db_dir};create=true",
        )
        .enableHiveSupport()
    )

    if extra_conf:
        for k, v in extra_conf.items():
            builder = builder.config(k, str(v))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

