from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from common.spark_session import build_spark


@dataclass(frozen=True)
class WarehouseOdsPaths:
    """
    Logical paths for standardized ODS landing tables (Parquet) under `warehouse/ods/`.

    Facts are dt-partitioned and should be filtered by `dt` for incremental processing.
    """

    warehouse_dir: Path

    def ad_event_log(self) -> Path:
        return self.warehouse_dir / "ods" / "ad_event_log"

    def conversion_log(self) -> Path:
        return self.warehouse_dir / "ods" / "conversion_log"

    def user_profile(self) -> Path:
        return self.warehouse_dir / "ods" / "user_profile"

    def ad_meta(self) -> Path:
        return self.warehouse_dir / "ods" / "ad_meta"

    def ad_slot(self) -> Path:
        return self.warehouse_dir / "ods" / "ad_slot"

    def ad_cost(self) -> Path:
        return self.warehouse_dir / "ods" / "ad_cost"

    def discover_dts(self) -> List[str]:
        """
        Discover dt partitions from ODS landing.

        Prefer filesystem discovery of `dt=...` directories to avoid full table scans.
        """
        base = self.ad_event_log()
        if not base.exists():
            return []
        dt_dirs = sorted([p for p in base.glob("dt=*") if p.is_dir()])
        return [p.name.replace("dt=", "", 1) for p in dt_dirs]


def _project_root() -> Path:
    """
    Resolve repo root directory.

    Assumption: this file lives at `<repo>/jobs/build_dwd.py`.
    """
    return Path(__file__).resolve().parents[1]


def _warehouse_table_path(warehouse_dir: Path, rel: str) -> Path:
    """
    Standardize physical paths for DWD outputs under `warehouse/`.

    Example: rel="dwd/dwd_ad_impression_detail" -> warehouse/dwd/dwd_ad_impression_detail
    """
    return warehouse_dir / rel

def _read_parquet(spark, path: Path) -> DataFrame:
    """Read a parquet table directory."""
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet path: {path}")
    return spark.read.parquet(str(path))


def _table_exists(spark, table_name: str) -> bool:
    """Guard catalog access so the job still works before Hive is fully wired in."""
    try:
        return spark.catalog.tableExists(table_name)
    except Exception:
        return False


def _read_table_or_parquet(spark, table_name: str, fallback_path: Path) -> DataFrame:
    """
    Prefer reading standardized Hive/Spark tables.

    We keep a parquet fallback so the project remains runnable before the whole
    Hive layer is fully connected.
    """
    if _table_exists(spark, table_name):
        return spark.table(table_name)
    return _read_parquet(spark, fallback_path)

def _read_ods_fact_for_dt(
    spark, warehouse_ods: WarehouseOdsPaths, dt: str
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Read ODS fact tables for a single dt from standardized ODS landing (Parquet).

    Inputs:
    - warehouse/ods/ad_event_log
    - warehouse/ods/conversion_log
    - warehouse/ods/ad_cost

    All are partitioned by dt, so we filter by dt for incremental processing.
    """
    event = _read_table_or_parquet(spark, "ods.ad_event_log", warehouse_ods.ad_event_log()).where(F.col("dt") == dt)
    conv = _read_table_or_parquet(spark, "ods.conversion_log", warehouse_ods.conversion_log()).where(F.col("dt") == dt)
    cost = _read_table_or_parquet(spark, "ods.ad_cost", warehouse_ods.ad_cost()).where(F.col("dt") == dt)
    return event, conv, cost

def _discover_dts(warehouse_ods: WarehouseOdsPaths) -> List[str]:
    """
    Discover dt values only from standardized ODS landing.

    DWD now depends on ODS Hive tables first, with `warehouse/ods/...` parquet
    as the only fallback. It no longer infers partitions from raw CSV folders.
    """
    return warehouse_ods.discover_dts()


def _ensure_cols(df: DataFrame, cols_with_default: dict[str, F.Column]) -> DataFrame:
    """
    Backward-compat helper.

    If upstream ODS schema evolves, this prevents the DWD job from breaking when
    older CSVs miss newly added columns.
    """
    out = df
    for c, default_expr in cols_with_default.items():
        if c not in out.columns:
            out = out.withColumn(c, default_expr)
    return out


def _sql_string(s: str) -> str:
    return s.replace("'", "''")


def _write_partitioned(df: DataFrame, out_path: Path) -> None:
    df.write.mode("overwrite").partitionBy("dt").parquet(str(out_path))


def _write_partitioned_table_or_path(
    spark,
    df: DataFrame,
    table_name: str,
    out_path: Path,
    dt: str,
    data_cols: list[str],
) -> None:
    """
    Prefer table-style writes when a catalog table already exists.

    This keeps the code close to a Hive/Spark SQL job style:
    - Python orchestrates the run
    - SQL/DataFrame results are inserted into named tables when possible

    If the target Hive table is not available yet, we keep the current Parquet
    output behavior so the project remains runnable.

    `data_cols` means the non-partition columns written to the target table.
    The SELECT order must match the Hive table schema order because dt is
    supplied separately via `PARTITION (dt='...')`.
    """
    if _table_exists(spark, table_name):
        temp_view = f"_stage_{table_name.replace('.', '_')}"
        select_cols = ", ".join(f"`{c}`" for c in data_cols)
        df.createOrReplaceTempView(temp_view)
        spark.sql(
            f"""
INSERT OVERWRITE TABLE {table_name} PARTITION (dt='{_sql_string(dt)}')
SELECT {select_cols}
FROM {temp_view}
WHERE dt = '{_sql_string(dt)}'
""".strip()
        )
        spark.catalog.dropTempView(temp_view)
        return

    _write_partitioned(df, out_path)


def _dq_print_event_uniqueness(df: DataFrame, dt: str, name: str) -> None:
    total = df.count()
    distinct = df.select("event_id").distinct().count() if total else 0
    rate = (distinct / total) if total else 1.0
    print(f"[DQ] {name} dt={dt} event_id uniqueness: {distinct}/{total} = {rate:.2%}")


def _dq_print_filter_effect(before: int, after: int, name: str, dt: str) -> None:
    print(
        f"[DQ] {name} dt={dt} is_valid filter: before={before:,}, after={after:,}, kept={(after / before):.2%}"
        if before
        else f"[DQ] {name} dt={dt} is_valid filter: before=0, after=0"
    )


def _dq_print_missing_rate(df: DataFrame, col: str, name: str, dt: str) -> None:
    total = df.count()
    missing = df.where(F.col(col).isNull()).count() if total else 0
    rate = (missing / total) if total else 0.0
    print(f"[DQ] {name} dt={dt} {col} missing rate: {missing}/{total} = {rate:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DWD detail tables + dimensions. Prefer ODS Hive tables; fallback to parquet during migration."
    )
    parser.add_argument(
        "--dt",
        default=None,
        help="Process single dt (YYYY-MM-DD). If omitted, process all dt discovered from warehouse/ods landing.",
    )
    parser.add_argument("--warehouse_dir", default=str(_project_root() / "warehouse"), help="Warehouse base dir (warehouse)")
    parser.add_argument("--no_dq", action="store_true", help="Skip DQ prints that trigger expensive counts.")
    args = parser.parse_args()

    warehouse_dir = Path(args.warehouse_dir)
    warehouse_ods = WarehouseOdsPaths(warehouse_dir=warehouse_dir)

    spark = build_spark(app_name="build_dwd")

    # Determine dt list:
    # - when --dt is provided: single-day incremental build
    # - otherwise: discover all dt partitions under standardized ODS landing
    dts: List[str]
    if args.dt:
        dts = [args.dt]
    else:
        dts = _discover_dts(warehouse_ods)
        if not dts:
            raise FileNotFoundError(
                "Cannot discover dt partitions. Ensure ODS landing exists at "
                f"{warehouse_ods.ad_event_log()} (partitioned dt=YYYY-MM-DD) "
                "or provide --dt explicitly."
            )

    # Python still orchestrates input resolution. Core modeling below is SQL-first.
    ods_user_profile = _read_table_or_parquet(spark, "ods.user_profile", warehouse_ods.user_profile())
    ods_ad_meta = _read_table_or_parquet(spark, "ods.ad_meta", warehouse_ods.ad_meta())
    ods_ad_slot = _read_table_or_parquet(spark, "ods.ad_slot", warehouse_ods.ad_slot())

    ods_user_profile.createOrReplaceTempView("ods_user_profile_src")
    ods_ad_meta.createOrReplaceTempView("ods_ad_meta_src")
    ods_ad_slot.createOrReplaceTempView("ods_ad_slot_src")

    # Design note:
    # - dwd.dim_ad is modeled at (ad_id, campaign_id) grain
    # - this matches the downstream fact enrich rule and avoids losing mappings
    #   when one ad_id appears under multiple campaigns
    # Build the user dimension by standardizing ODS user profile fields.
    # The current ODS generator guarantees 1 row per user_id, so no extra dedup is needed here.
    sql_dim_user = """
SELECT
  CAST(user_id AS STRING) AS user_id,
  CAST(gender AS STRING) AS gender,
  CAST(age AS INT) AS age,
  CAST(region AS STRING) AS region,
  CAST(device_type AS STRING) AS device_type,
  TO_DATE(CAST(register_dt AS STRING), 'yyyy-MM-dd') AS register_dt
FROM ods_user_profile_src
""".strip()

    # Build the ad dimension at (ad_id, campaign_id) grain from ODS ad metadata.
    # The current ODS generator guarantees 1 row per (ad_id, campaign_id), so we only standardize fields here.
    sql_dim_ad = """
SELECT
  CAST(ad_id AS STRING) AS ad_id,
  CAST(campaign_id AS STRING) AS campaign_id,
  CAST(advertiser_id AS STRING) AS advertiser_id,
  CAST(ad_type AS STRING) AS ad_type,
  CAST(landing_type AS STRING) AS landing_type,
  CASE
    WHEN product_id IS NULL OR CAST(product_id AS STRING) = '' THEN NULL
    ELSE CAST(product_id AS STRING)
  END AS product_id,
  TO_DATE(CAST(start_dt AS STRING), 'yyyy-MM-dd') AS start_dt,
  TO_DATE(CAST(end_dt AS STRING), 'yyyy-MM-dd') AS end_dt
FROM ods_ad_meta_src
""".strip()

    # Build the ad slot dimension by standardizing slot attributes.
    # The current ODS generator guarantees 1 row per ad_slot_id, so no extra dedup is needed here.
    sql_dim_ad_slot = """
SELECT
  CAST(ad_slot_id AS STRING) AS ad_slot_id,
  CAST(slot_type AS STRING) AS slot_type,
  CAST(app AS STRING) AS app,
  CAST(position AS STRING) AS position,
  CAST(price_factor AS DOUBLE) AS price_factor
FROM ods_ad_slot_src
""".strip()

    # Let spark execute the sql(Spark SQL), it will return a DataFrame
    dwd_dim_user_base = spark.sql(sql_dim_user)
    dwd_dim_ad_base = spark.sql(sql_dim_ad)
    dwd_dim_ad_slot_base = spark.sql(sql_dim_ad_slot)

    # Create temporary views for the base dimensions.
    dwd_dim_user_base.createOrReplaceTempView("dwd_dim_user_base_v")
    dwd_dim_ad_base.createOrReplaceTempView("dwd_dim_ad_base_v")
    dwd_dim_ad_slot_base.createOrReplaceTempView("dwd_dim_ad_slot_base_v")

    # Dimensions landing strategy:
    # - write as "daily snapshot" partitioned by dt
    # - makes partitioning consistent with facts and supports incremental reruns by dt
    for dt in dts:
        dim_user = dwd_dim_user_base.withColumn("dt", F.lit(dt))
        dim_ad = dwd_dim_ad_base.withColumn("dt", F.lit(dt))
        dim_slot = dwd_dim_ad_slot_base.withColumn("dt", F.lit(dt))

        _write_partitioned_table_or_path(
            spark,
            dim_user,
            "dwd.dim_user",
            _warehouse_table_path(warehouse_dir, "dwd/dim/dwd_dim_user"),
            dt,
            ["user_id", "gender", "age", "region", "device_type", "register_dt"],
        )
        _write_partitioned_table_or_path(
            spark,
            dim_ad,
            "dwd.dim_ad",
            _warehouse_table_path(warehouse_dir, "dwd/dim/dwd_dim_ad"),
            dt,
            ["ad_id", "campaign_id", "advertiser_id", "ad_type", "landing_type", "product_id", "start_dt", "end_dt"],
        )
        _write_partitioned_table_or_path(
            spark,
            dim_slot,
            "dwd.dim_ad_slot",
            _warehouse_table_path(warehouse_dir, "dwd/dim/dwd_dim_ad_slot"),
            dt,
            ["ad_slot_id", "slot_type", "app", "position", "price_factor"],
        )

    # Build fact/detail tables per dt
    for dt in dts:
        ods_event, ods_conv, ods_cost = _read_ods_fact_for_dt(spark, warehouse_ods, dt)
        
        # Keep backward compatibility if an older ODS event snapshot misses newly added raw fields.
        ods_event = _ensure_cols(
            ods_event,
            {
                "session_id": F.lit(None).cast("string"),
                "ad_slot_id": F.lit(None).cast("string"),
                "page_id": F.lit(None).cast("string"),
                "is_valid": F.lit(1).cast("int"),
                "ip": F.lit(None).cast("string"),
                "user_agent": F.lit(None).cast("string"),
            },
        )

        # Create temporary views for the ODS fact tables.
        ods_event.createOrReplaceTempView("ods_event_src")
        ods_conv.createOrReplaceTempView("ods_conv_src")
        ods_cost.createOrReplaceTempView("ods_cost_src")

        # -------------------- SQL definitions --------------------
        # Standardize ODS event fields into typed columns.
        sql_event_prepared = f"""
SELECT
  CAST(event_id AS STRING) AS event_id,
  TO_TIMESTAMP(CAST(event_time AS STRING), 'yyyy-MM-dd HH:mm:ss') AS event_time,
  CAST(dt AS STRING) AS dt,
  CAST(user_id AS STRING) AS user_id,
  CAST(session_id AS STRING) AS session_id,
  CAST(ad_id AS STRING) AS ad_id,
  CAST(campaign_id AS STRING) AS campaign_id,
  CAST(creative_id AS STRING) AS creative_id,
  CAST(site_id AS STRING) AS site_id,
  CAST(page_id AS STRING) AS page_id,
  CAST(ad_slot_id AS STRING) AS ad_slot_id,
  CAST(device AS STRING) AS device,
  CAST(user_agent AS STRING) AS user_agent,
  CAST(ip AS STRING) AS ip,
  CAST(is_valid AS INT) AS is_valid,
  CAST(event_type AS STRING) AS event_type
FROM ods_event_src
WHERE dt = '{_sql_string(dt)}'
""".strip()

        # Standardize ODS conversion fields into typed columns.
        sql_conv_prepared = f"""
SELECT
  CAST(conv_id AS STRING) AS conv_id,
  TO_TIMESTAMP(CAST(conv_time AS STRING), 'yyyy-MM-dd HH:mm:ss') AS conv_time,
  CAST(dt AS STRING) AS dt,
  CAST(user_id AS STRING) AS user_id,
  CAST(ad_id AS STRING) AS ad_id,
  CAST(campaign_id AS STRING) AS campaign_id,
  CAST(order_id AS STRING) AS order_id,
  CAST(gmv_amount AS DOUBLE) AS gmv_amount
FROM ods_conv_src
WHERE dt = '{_sql_string(dt)}'
""".strip()

        # Standardize ODS daily campaign cost.
        sql_cost_prepared = f"""
SELECT
  CAST(dt AS STRING) AS dt,
  CAST(campaign_id AS STRING) AS campaign_id,
  CAST(cost AS DOUBLE) AS cost
FROM ods_cost_src
WHERE dt = '{_sql_string(dt)}'
""".strip()

        # Enrich raw event traffic with ad metadata and ad slot attributes to form a wide event base.
        # Broadcast join is used to avoid shuffling the data.
        sql_base_event = f"""
SELECT /*+ BROADCAST(m), BROADCAST(s) */
  e.event_id,
  e.event_time,
  e.dt,
  e.user_id,
  e.session_id,
  e.ad_id,
  e.campaign_id,
  e.creative_id,
  e.site_id,
  e.page_id,
  e.ad_slot_id,
  e.device,
  e.user_agent,
  e.ip,
  e.is_valid,
  e.event_type,
  m.advertiser_id,
  m.ad_type,
  m.landing_type,
  m.product_id,
  s.slot_type,
  s.app,
  s.position,
  s.price_factor
FROM ods_event_prepared_v e
LEFT JOIN dwd_dim_ad_base_v m
  ON e.ad_id = m.ad_id AND e.campaign_id = m.campaign_id
LEFT JOIN dwd_dim_ad_slot_base_v s
  ON e.ad_slot_id = s.ad_slot_id
""".strip()

        # Filter impression traffic, keep valid rows only, and deduplicate by (dt, event_id).
        sql_impression_detail = """
SELECT
  event_id,
  event_time,
  dt,
  user_id,
  session_id,
  ad_id,
  campaign_id,
  creative_id,
  advertiser_id,
  ad_type,
  landing_type,
  product_id,
  site_id,
  page_id,
  ad_slot_id,
  slot_type,
  app,
  position,
  price_factor,
  device,
  user_agent,
  ip,
  event_type
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY dt, event_id ORDER BY event_time DESC) AS rn
  FROM base_event_v
  WHERE event_type = 'impression' AND COALESCE(is_valid, 1) = 1
) t
WHERE rn = 1
""".strip()

        # Filter click traffic, keep valid rows only, and deduplicate by (dt, event_id).
        sql_click_detail = """
SELECT
  event_id,
  event_time,
  dt,
  user_id,
  session_id,
  ad_id,
  campaign_id,
  creative_id,
  advertiser_id,
  ad_type,
  landing_type,
  product_id,
  site_id,
  page_id,
  ad_slot_id,
  slot_type,
  app,
  position,
  price_factor,
  device,
  user_agent,
  ip,
  event_type
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY dt, event_id ORDER BY event_time DESC) AS rn
  FROM base_event_v
  WHERE event_type = 'click' AND COALESCE(is_valid, 1) = 1
) t
WHERE rn = 1
""".strip()

        # Conversion detail
        # Note: conversion table has no is_valid flag in this project; we only dedup and enrich.
        # Build conversion detail by joining conversion logs to ad metadata and deduplicating by (dt, conv_id).
        sql_conversion_detail = """
SELECT
  conv_id,
  conv_time,
  dt,
  user_id,
  ad_id,
  campaign_id,
  advertiser_id,
  product_id,
  order_id,
  gmv_amount
FROM (
  SELECT
    c.conv_id,
    c.conv_time,
    c.dt,
    c.user_id,
    c.ad_id,
    c.campaign_id,
    m.advertiser_id,
    m.product_id,
    c.order_id,
    c.gmv_amount,
    ROW_NUMBER() OVER (PARTITION BY c.dt, c.conv_id ORDER BY c.conv_time DESC) AS rn
  FROM ods_conv_prepared_v c
  LEFT JOIN dwd_dim_ad_base_v m
    ON c.ad_id = m.ad_id AND c.campaign_id = m.campaign_id
) t
WHERE rn = 1
""".strip()

        # Build the DWD campaign-day cost table from standardized ODS cost input at (dt, campaign_id) grain.
        sql_campaign_day_cost = """
SELECT
  campaign_id,
  cost,
  dt
FROM (
  SELECT
    campaign_id,
    cost,
    dt,
    ROW_NUMBER() OVER (
      PARTITION BY dt, campaign_id
      ORDER BY cost DESC, campaign_id
    ) AS rn
  FROM ods_cost_prepared_v
  WHERE campaign_id IS NOT NULL
) t
WHERE rn = 1
""".strip()

        # -------------------- SQL execution --------------------
        spark.sql(sql_event_prepared).createOrReplaceTempView("ods_event_prepared_v")
        spark.sql(sql_conv_prepared).createOrReplaceTempView("ods_conv_prepared_v")
        spark.sql(sql_cost_prepared).createOrReplaceTempView("ods_cost_prepared_v")
        spark.sql(sql_base_event).createOrReplaceTempView("base_event_v")

        imp_before = spark.sql("SELECT * FROM base_event_v WHERE event_type = 'impression'")
        clk_before = spark.sql("SELECT * FROM base_event_v WHERE event_type = 'click'")
        imp = spark.sql(sql_impression_detail)
        clk = spark.sql(sql_click_detail)
        conv_base = spark.sql(sql_conversion_detail)
        campaign_day_cost = spark.sql(sql_campaign_day_cost)

        # -------------------- DQ + writes --------------------
        if not args.no_dq:
            imp_before_cnt = imp_before.count()
            imp_after_cnt = imp.count()
            _dq_print_event_uniqueness(imp, dt, "dwd_ad_impression_detail")
            _dq_print_filter_effect(imp_before_cnt, imp_after_cnt, "dwd_ad_impression_detail", dt)
            _dq_print_missing_rate(imp, "advertiser_id", "dwd_ad_impression_detail", dt)
            _dq_print_missing_rate(imp, "slot_type", "dwd_ad_impression_detail(ad_slot join)", dt)
            adv_cnt = imp.select("advertiser_id").distinct().count()
            print(f"[DQ] dwd_ad_impression_detail dt={dt} distinct advertiser_id: {adv_cnt}")

        imp_out = _warehouse_table_path(warehouse_dir, "dwd/dwd_ad_impression_detail")
        _write_partitioned_table_or_path(
            spark,
            imp,
            "dwd.ad_impression_detail",
            imp_out,
            dt,
            [
                "event_id",
                "event_time",
                "user_id",
                "session_id",
                "ad_id",
                "campaign_id",
                "creative_id",
                "advertiser_id",
                "ad_type",
                "landing_type",
                "product_id",
                "site_id",
                "page_id",
                "ad_slot_id",
                "slot_type",
                "app",
                "position",
                "price_factor",
                "device",
                "user_agent",
                "ip",
                "event_type",
            ],
        )

        if not args.no_dq:
            clk_before_cnt = clk_before.count()
            clk_after_cnt = clk.count()
            _dq_print_event_uniqueness(clk, dt, "dwd_ad_click_detail")
            _dq_print_filter_effect(clk_before_cnt, clk_after_cnt, "dwd_ad_click_detail", dt)
            _dq_print_missing_rate(clk, "advertiser_id", "dwd_ad_click_detail", dt)
            _dq_print_missing_rate(clk, "slot_type", "dwd_ad_click_detail(ad_slot join)", dt)

        clk_out = _warehouse_table_path(warehouse_dir, "dwd/dwd_ad_click_detail")
        _write_partitioned_table_or_path(
            spark,
            clk,
            "dwd.ad_click_detail",
            clk_out,
            dt,
            [
                "event_id",
                "event_time",
                "user_id",
                "session_id",
                "ad_id",
                "campaign_id",
                "creative_id",
                "advertiser_id",
                "ad_type",
                "landing_type",
                "product_id",
                "site_id",
                "page_id",
                "ad_slot_id",
                "slot_type",
                "app",
                "position",
                "price_factor",
                "device",
                "user_agent",
                "ip",
                "event_type",
            ],
        )

        conv_out = _warehouse_table_path(warehouse_dir, "dwd/dwd_ad_conversion_detail")
        _write_partitioned_table_or_path(
            spark,
            conv_base,
            "dwd.ad_conversion_detail",
            conv_out,
            dt,
            [
                "conv_id",
                "conv_time",
                "user_id",
                "ad_id",
                "campaign_id",
                "advertiser_id",
                "product_id",
                "order_id",
                "gmv_amount",
            ],
        )

        if not args.no_dq:
            _dq_print_missing_rate(conv_base, "advertiser_id", "dwd_ad_conversion_detail", dt)

        cost_out = _warehouse_table_path(warehouse_dir, "dwd/dwd_campaign_day_cost")
        _write_partitioned_table_or_path(
            spark,
            campaign_day_cost,
            "dwd.campaign_day_cost",
            cost_out,
            dt,
            ["campaign_id", "cost"],
        )

    spark.stop()


if __name__ == "__main__":
    main()

