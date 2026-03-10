from __future__ import annotations

"""
Ingest ODS dimension/config CSVs into Parquet and register them in Hive.

This keeps the current lightweight CSV source files, but makes DWD depend on
formal ODS tables instead of directly depending on raw files.
"""

import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from common.spark_session import build_spark


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _warehouse_table_path(warehouse_dir: Path, layer: str, table: str) -> Path:
    return warehouse_dir / layer / table


def _read_csv(spark, path: Path, schema: StructType) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing ODS CSV: {path}")
    return spark.read.option("header", "true").schema(schema).csv(str(path))


def _sql_string(s: str) -> str:
    return s.replace("'", "''")


def _ensure_non_partitioned_ods_table(
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
LOCATION '{_sql_string(str(location))}'
""".strip()
    )
    print(f"[HIVE] Ensured table {table_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ODS dim/config CSVs into Parquet and Hive.")
    parser.add_argument("--input_dir", default=str(_project_root() / "data" / "ods"), help="ODS CSV directory")
    parser.add_argument("--warehouse_dir", default=str(_project_root() / "warehouse"), help="Warehouse base dir")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    warehouse_dir = Path(args.warehouse_dir)

    spark = build_spark(app_name="ingest_ods_dims")

    user_profile_schema = StructType(
        [
            StructField("user_id", StringType(), True),
            StructField("gender", StringType(), True),
            StructField("age", StringType(), True),
            StructField("region", StringType(), True),
            StructField("device_type", StringType(), True),
            StructField("register_dt", StringType(), True),
        ]
    )
    ad_meta_schema = StructType(
        [
            StructField("ad_id", StringType(), True),
            StructField("campaign_id", StringType(), True),
            StructField("advertiser_id", StringType(), True),
            StructField("ad_type", StringType(), True),
            StructField("landing_type", StringType(), True),
            StructField("product_id", StringType(), True),
            StructField("start_dt", StringType(), True),
            StructField("end_dt", StringType(), True),
        ]
    )
    ad_slot_schema = StructType(
        [
            StructField("ad_slot_id", StringType(), True),
            StructField("slot_type", StringType(), True),
            StructField("app", StringType(), True),
            StructField("position", StringType(), True),
            StructField("price_factor", StringType(), True),
        ]
    )

    user_profile = (
        _read_csv(spark, input_dir / "ods_user_profile.csv", user_profile_schema)
        .withColumn("user_id", F.col("user_id").cast("string"))
        .withColumn("gender", F.col("gender").cast("string"))
        .withColumn("age", F.col("age").cast(IntegerType()))
        .withColumn("region", F.col("region").cast("string"))
        .withColumn("device_type", F.col("device_type").cast("string"))
        .withColumn("register_dt", F.to_date(F.col("register_dt").cast("string"), "yyyy-MM-dd"))
    )
    ad_meta = (
        _read_csv(spark, input_dir / "ods_ad_meta.csv", ad_meta_schema)
        .withColumn("ad_id", F.col("ad_id").cast("string"))
        .withColumn("campaign_id", F.col("campaign_id").cast("string"))
        .withColumn("advertiser_id", F.col("advertiser_id").cast("string"))
        .withColumn("ad_type", F.col("ad_type").cast("string"))
        .withColumn("landing_type", F.col("landing_type").cast("string"))
        .withColumn(
            "product_id",
            F.when(F.col("product_id").isNull() | (F.col("product_id") == ""), F.lit(None))
            .otherwise(F.col("product_id"))
            .cast("string"),
        )
        .withColumn("start_dt", F.to_date(F.col("start_dt").cast("string"), "yyyy-MM-dd"))
        .withColumn("end_dt", F.to_date(F.col("end_dt").cast("string"), "yyyy-MM-dd"))
    )
    ad_slot = (
        _read_csv(spark, input_dir / "ods_ad_slot.csv", ad_slot_schema)
        .withColumn("ad_slot_id", F.col("ad_slot_id").cast("string"))
        .withColumn("slot_type", F.col("slot_type").cast("string"))
        .withColumn("app", F.col("app").cast("string"))
        .withColumn("position", F.col("position").cast("string"))
        .withColumn("price_factor", F.col("price_factor").cast(DoubleType()))
    )

    user_profile_out = _warehouse_table_path(warehouse_dir, "ods", "user_profile")
    ad_meta_out = _warehouse_table_path(warehouse_dir, "ods", "ad_meta")
    ad_slot_out = _warehouse_table_path(warehouse_dir, "ods", "ad_slot")

    user_profile.write.mode("overwrite").parquet(str(user_profile_out))
    ad_meta.write.mode("overwrite").parquet(str(ad_meta_out))
    ad_slot.write.mode("overwrite").parquet(str(ad_slot_out))

    _ensure_non_partitioned_ods_table(
        spark,
        "ods.user_profile",
        """
user_id STRING,
gender STRING,
age INT,
region STRING,
device_type STRING,
register_dt DATE
""".strip(),
        user_profile_out,
    )
    _ensure_non_partitioned_ods_table(
        spark,
        "ods.ad_meta",
        """
ad_id STRING,
campaign_id STRING,
advertiser_id STRING,
ad_type STRING,
landing_type STRING,
product_id STRING,
start_dt DATE,
end_dt DATE
""".strip(),
        ad_meta_out,
    )
    _ensure_non_partitioned_ods_table(
        spark,
        "ods.ad_slot",
        """
ad_slot_id STRING,
slot_type STRING,
app STRING,
position STRING,
price_factor DOUBLE
""".strip(),
        ad_slot_out,
    )

    spark.stop()


if __name__ == "__main__":
    main()

