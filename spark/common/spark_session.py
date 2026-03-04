from __future__ import annotations

from typing import Any, Dict, Optional

from pyspark.sql import SparkSession


def build_spark(app_name: str, extra_conf: Optional[Dict[str, Any]] = None) -> SparkSession:
    """
    Unified SparkSession config for local offline DW jobs.

    Notes:
    - Local mode, Parquet storage
    - AQE enabled (common interview point in China big-data roles)
    - Dynamic partition overwrite to mimic Hive dt=... partitions
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
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
    )

    if extra_conf:
        for k, v in extra_conf.items():
            builder = builder.config(k, str(v))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

