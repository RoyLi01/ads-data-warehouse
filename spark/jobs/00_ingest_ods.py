from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from spark.common.spark_session import build_spark


def _project_root() -> Path:
    # spark/jobs/*.py -> spark/jobs -> spark -> project root
    return Path(__file__).resolve().parents[2]


def _warehouse_table_path(warehouse_dir: Path, layer: str, table: str) -> Path:
    return warehouse_dir / layer / table


def _read_csv(spark, path: Path, schema: StructType) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing ODS CSV: {path}")
    return spark.read.option("header", "true").schema(schema).csv(str(path))


def main() -> None:
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

    input_dir = Path(args.input_dir)
    warehouse_dir = Path(args.warehouse_dir)

    spark = build_spark(app_name="00_ingest_ods")

    # ------------------- ods_ad_event_log -------------------
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
        ]
    )
    event_csv = input_dir / "ods_ad_event_log.csv"
    ad_event_log = _read_csv(spark, event_csv, event_schema).withColumn(
        "event_time", F.to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss")
    )
    (
        ad_event_log.repartition(F.col("dt"))
        .write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(_warehouse_table_path(warehouse_dir, "ods", "ad_event_log")))
    )

    # ------------------- ods_conversion_log -------------------
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

    # ------------------- ods_ad_cost -------------------
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

    spark.stop()


if __name__ == "__main__":
    main()

