from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from spark.common.spark_session import build_spark


@dataclass(frozen=True)
class OdsPaths:
    base_dir: Path

    def dt_dir(self, dt: str) -> Path:
        return self.base_dir / f"dt={dt}"

    def root_event_csv(self) -> Path:
        return self.base_dir / "ods_ad_event_log.csv"

    def root_conv_csv(self) -> Path:
        return self.base_dir / "ods_conversion_log.csv"

    def root_cost_csv(self) -> Path:
        return self.base_dir / "ods_ad_cost.csv"

    def root_user_profile_csv(self) -> Path:
        return self.base_dir / "ods_user_profile.csv"

    def root_ad_meta_csv(self) -> Path:
        return self.base_dir / "ods_ad_meta.csv"

    def root_ad_slot_csv(self) -> Path:
        return self.base_dir / "ods_ad_slot.csv"


def _project_root() -> Path:
    # warehouse/*.py -> warehouse -> project root
    return Path(__file__).resolve().parents[1]


def _warehouse_table_path(warehouse_dir: Path, rel: str) -> Path:
    return warehouse_dir / rel


def _read_csv(spark, path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return spark.read.option("header", "true").csv(str(path))


def _read_ods_fact_for_dt(spark, ods: OdsPaths, dt: str) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Read ODS fact CSVs for a single dt.

    Priority:
    1) data/ods/dt=YYYY-MM-DD/*.csv if exists
    2) fallback to root CSV (data/ods/*.csv) and filter by dt
    """
    dt_dir = ods.dt_dir(dt)
    if dt_dir.exists():
        event = _read_csv(spark, dt_dir / "ods_ad_event_log.csv")
        conv = _read_csv(spark, dt_dir / "ods_conversion_log.csv")
        cost = _read_csv(spark, dt_dir / "ods_ad_cost.csv")
        return event, conv, cost

    event = _read_csv(spark, ods.root_event_csv()).where(F.col("dt") == dt)
    conv = _read_csv(spark, ods.root_conv_csv()).where(F.col("dt") == dt)
    cost = _read_csv(spark, ods.root_cost_csv()).where(F.col("dt") == dt)
    return event, conv, cost


def _discover_dts(ods: OdsPaths) -> List[str]:
    """
    Discover dt values to process.
    - Prefer dt=YYYY-MM-DD directories (if present)
    - Else infer distinct dt from root event CSV
    """
    dt_dirs = sorted([p for p in ods.base_dir.glob("dt=*") if p.is_dir()])
    if dt_dirs:
        return [p.name.replace("dt=", "", 1) for p in dt_dirs]

    # fallback: read root event csv via pandas-like parsing is avoided; use Spark in main() instead
    return []


def _ensure_cols(df: DataFrame, cols_with_default: dict[str, F.Column]) -> DataFrame:
    out = df
    for c, default_expr in cols_with_default.items():
        if c not in out.columns:
            out = out.withColumn(c, default_expr)
    return out


def _standardize_event_time(df: DataFrame, col: str) -> DataFrame:
    return df.withColumn(col, F.to_timestamp(F.col(col).cast("string"), "yyyy-MM-dd HH:mm:ss"))


def _dedup(df: DataFrame, keys: list[str]) -> DataFrame:
    return df.dropDuplicates(keys)


def _dq_print_event_uniqueness(df: DataFrame, dt: str, name: str) -> None:
    total = df.count()
    distinct = df.select("event_id").distinct().count() if total else 0
    rate = (distinct / total) if total else 1.0
    print(f"[DQ] {name} dt={dt} event_id uniqueness: {distinct}/{total} = {rate:.2%}")


def _dq_print_filter_effect(before: int, after: int, name: str, dt: str) -> None:
    print(f"[DQ] {name} dt={dt} is_valid filter: before={before:,}, after={after:,}, kept={(after / before):.2%}" if before else f"[DQ] {name} dt={dt} is_valid filter: before=0, after=0")


def _dq_print_missing_rate(df: DataFrame, col: str, name: str, dt: str) -> None:
    total = df.count()
    missing = df.where(F.col(col).isNull()).count() if total else 0
    rate = (missing / total) if total else 0.0
    print(f"[DQ] {name} dt={dt} {col} missing rate: {missing}/{total} = {rate:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DWD v2 detail tables + dimensions from ODS v2 CSVs.")
    parser.add_argument("--dt", default=None, help="Process single dt (YYYY-MM-DD). If omitted, process all dt in data/ods.")
    parser.add_argument("--ods_dir", default=str(_project_root() / "data" / "ods"), help="ODS CSV base dir (data/ods)")
    parser.add_argument("--warehouse_dir", default=str(_project_root() / "warehouse"), help="Warehouse base dir (warehouse)")
    args = parser.parse_args()

    ods = OdsPaths(base_dir=Path(args.ods_dir))
    warehouse_dir = Path(args.warehouse_dir)

    spark = build_spark(app_name="warehouse_01_build_dwd_v2")

    # Determine dt list
    dts: List[str]
    if args.dt:
        dts = [args.dt]
    else:
        dts = _discover_dts(ods)
        if not dts:
            # fallback: infer from root event csv
            root_event = _read_csv(spark, ods.root_event_csv())
            dts = [r["dt"] for r in root_event.select("dt").distinct().collect()]
            dts.sort()

    # Read dimension/config CSVs (root only)
    ods_user_profile = _read_csv(spark, ods.root_user_profile_csv())
    ods_ad_meta = _read_csv(spark, ods.root_ad_meta_csv())
    ods_ad_slot = _read_csv(spark, ods.root_ad_slot_csv())

    # Normalize and deduplicate dimensions once
    dwd_dim_user_base = (
        ods_user_profile.select(
            F.col("user_id").cast("string").alias("user_id"),
            F.col("gender").cast("string").alias("gender"),
            F.col("age").cast(IntegerType()).alias("age"),
            F.col("region").cast("string").alias("region"),
            F.col("device_type").cast("string").alias("device_type"),
            F.to_date(F.col("register_dt").cast("string"), "yyyy-MM-dd").alias("register_dt"),
        )
        .dropDuplicates(["user_id"])
    )

    # NOTE:
    # - ods_ad_meta is generated from distinct (ad_id, campaign_id) pairs, so one ad_id can appear under multiple campaigns.
    # - If we de-duplicate only by ad_id, we may accidentally bias all ad_id to the hot campaign (first occurrence),
    #   causing advertiser_id to collapse to a single value.
    # - For v2 fact enrichment we therefore join by (ad_id, campaign_id) to preserve correct advertiser mapping.
    dwd_dim_ad_base = (
        ods_ad_meta.select(
            F.col("ad_id").cast("string").alias("ad_id"),
            F.col("campaign_id").cast("string").alias("campaign_id"),
            F.col("advertiser_id").cast("string").alias("advertiser_id"),
            F.col("ad_type").cast("string").alias("ad_type"),
            F.col("landing_type").cast("string").alias("landing_type"),
            F.when(F.col("product_id").isNull() | (F.col("product_id") == ""), F.lit(None)).otherwise(F.col("product_id")).cast("string").alias("product_id"),
            F.to_date(F.col("start_dt").cast("string"), "yyyy-MM-dd").alias("start_dt"),
            F.to_date(F.col("end_dt").cast("string"), "yyyy-MM-dd").alias("end_dt"),
        )
        .dropDuplicates(["ad_id"])
    )

    meta_pair_base = (
        ods_ad_meta.select(
            F.col("ad_id").cast("string").alias("ad_id"),
            F.col("campaign_id").cast("string").alias("campaign_id"),
            F.col("advertiser_id").cast("string").alias("advertiser_id"),
            F.col("ad_type").cast("string").alias("ad_type"),
            F.col("landing_type").cast("string").alias("landing_type"),
            F.when(F.col("product_id").isNull() | (F.col("product_id") == ""), F.lit(None))
            .otherwise(F.col("product_id"))
            .cast("string")
            .alias("product_id"),
        )
        .dropDuplicates(["ad_id", "campaign_id"])
    )

    dwd_dim_ad_slot_base = (
        ods_ad_slot.select(
            F.col("ad_slot_id").cast("string").alias("ad_slot_id"),
            F.col("slot_type").cast("string").alias("slot_type"),
            F.col("app").cast("string").alias("app"),
            F.col("position").cast("string").alias("position"),
            F.col("price_factor").cast(DoubleType()).alias("price_factor"),
        )
        .dropDuplicates(["ad_slot_id"])
    )

    # Dimensions landing strategy: "daily snapshot" partitioned by dt.
    # Reason: keep the same dt-partition convention as facts; supports point-in-time joins and incremental dt runs.
    for dt in dts:
        dim_user = dwd_dim_user_base.withColumn("dt", F.lit(dt))
        dim_ad = dwd_dim_ad_base.withColumn("dt", F.lit(dt))
        dim_slot = dwd_dim_ad_slot_base.withColumn("dt", F.lit(dt))

        (
            dim_user.write.mode("overwrite")
            .partitionBy("dt")
            .parquet(str(_warehouse_table_path(warehouse_dir, "dwd/dim/dwd_dim_user")))
        )
        (
            dim_ad.write.mode("overwrite")
            .partitionBy("dt")
            .parquet(str(_warehouse_table_path(warehouse_dir, "dwd/dim/dwd_dim_ad")))
        )
        (
            dim_slot.write.mode("overwrite")
            .partitionBy("dt")
            .parquet(str(_warehouse_table_path(warehouse_dir, "dwd/dim/dwd_dim_ad_slot")))
        )

    # Build v2 detail tables per dt
    for dt in dts:
        ods_event, ods_conv, _ods_cost = _read_ods_fact_for_dt(spark, ods, dt)

        # Ensure v2 columns exist (if source is old, default to keep compatibility)
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

        ods_event = _standardize_event_time(ods_event, "event_time")
        ods_conv = _standardize_event_time(ods_conv, "conv_time").withColumn("gmv_amount", F.col("gmv_amount").cast("double"))

        # ---------------- v2 impression/click (filter is_valid=1 only for v2) ----------------
        base = (
            ods_event.where(F.col("dt") == dt)
            .select(
                "event_id",
                "event_time",
                "dt",
                "user_id",
                "session_id",
                "ad_id",
                "campaign_id",
                "creative_id",
                "site_id",
                "page_id",
                "ad_slot_id",
                "device",
                "user_agent",
                "ip",
                "is_valid",
                "event_type",
            )
        )

        # join dims (left join, keep facts)
        # v2 fact enrichment uses (ad_id, campaign_id) to avoid advertiser collapse.
        base = (
            base.join(
                F.broadcast(meta_pair_base),
                on=["ad_id", "campaign_id"],
                how="left",
            )
            .join(F.broadcast(dwd_dim_ad_slot_base), on="ad_slot_id", how="left")
        )

        # Impression v2
        imp_before = base.where(F.col("event_type") == "impression")
        imp_before_cnt = imp_before.count()
        imp = (
            imp_before.where(F.col("is_valid").cast("int") == 1)
            .drop("is_valid")
        )
        imp_after_cnt = imp.count()
        imp = _dedup(imp, ["dt", "event_id"])

        # DQ prints
        _dq_print_event_uniqueness(imp, dt, "dwd_ad_impression_detail_v2")
        _dq_print_filter_effect(imp_before_cnt, imp_after_cnt, "dwd_ad_impression_detail_v2", dt)
        _dq_print_missing_rate(imp, "advertiser_id", "dwd_ad_impression_detail_v2", dt)
        _dq_print_missing_rate(imp, "slot_type", "dwd_ad_impression_detail_v2(ad_slot join)", dt)
        adv_cnt = imp.select("advertiser_id").distinct().count()
        print(f"[DQ] dwd_ad_impression_detail_v2 dt={dt} distinct advertiser_id: {adv_cnt}")

        imp_out = _warehouse_table_path(warehouse_dir, "dwd/dwd_ad_impression_detail_v2")
        imp.select(
            "event_id",
            "event_time",
            "dt",
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
        ).write.mode("overwrite").partitionBy("dt").parquet(str(imp_out))

        # Click v2
        clk_before = base.where(F.col("event_type") == "click")
        clk_before_cnt = clk_before.count()
        clk = (
            clk_before.where(F.col("is_valid").cast("int") == 1)
            .drop("is_valid")
        )
        clk_after_cnt = clk.count()
        clk = _dedup(clk, ["dt", "event_id"])

        _dq_print_event_uniqueness(clk, dt, "dwd_ad_click_detail_v2")
        _dq_print_filter_effect(clk_before_cnt, clk_after_cnt, "dwd_ad_click_detail_v2", dt)
        _dq_print_missing_rate(clk, "advertiser_id", "dwd_ad_click_detail_v2", dt)
        _dq_print_missing_rate(clk, "slot_type", "dwd_ad_click_detail_v2(ad_slot join)", dt)

        clk_out = _warehouse_table_path(warehouse_dir, "dwd/dwd_ad_click_detail_v2")
        clk.select(
            "event_id",
            "event_time",
            "dt",
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
        ).write.mode("overwrite").partitionBy("dt").parquet(str(clk_out))

        # ---------------- v2 conversion ----------------
        conv_base = (
            ods_conv.where(F.col("dt") == dt)
            .select("conv_id", "conv_time", "dt", "user_id", "ad_id", "campaign_id", "order_id", "gmv_amount")
        )
        conv_base = _dedup(conv_base, ["dt", "conv_id"])
        conv_base = conv_base.join(
            F.broadcast(meta_pair_base.select("ad_id", "campaign_id", "advertiser_id", "product_id")),
            on=["ad_id", "campaign_id"],
            how="left",
        )
        conv_out = _warehouse_table_path(warehouse_dir, "dwd/dwd_ad_conversion_detail_v2")
        conv_base.select(
            "conv_id",
            "conv_time",
            "dt",
            "user_id",
            "ad_id",
            "campaign_id",
            "advertiser_id",
            "product_id",
            "order_id",
            "gmv_amount",
        ).write.mode("overwrite").partitionBy("dt").parquet(str(conv_out))

        _dq_print_missing_rate(conv_base, "advertiser_id", "dwd_ad_conversion_detail_v2", dt)

    spark.stop()


if __name__ == "__main__":
    main()

