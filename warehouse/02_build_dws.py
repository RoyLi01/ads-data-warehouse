from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
import sys

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark.common.spark_session import build_spark


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def warehouse_dir(self) -> Path:
        return self.project_root / "warehouse"

    @property
    def ods_dir(self) -> Path:
        return self.project_root / "data" / "ods"

    def ods_dt_dir(self, dt: str) -> Path:
        return self.ods_dir / f"dt={dt}"

    def warehouse_table(self, rel: str) -> Path:
        return self.warehouse_dir / rel


def _project_root() -> Path:
    # warehouse/*.py -> warehouse -> project root
    return Path(__file__).resolve().parents[1]


def _read_parquet(spark, path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet path: {path}")
    return spark.read.parquet(str(path))


def _read_csv(spark, path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return spark.read.option("header", "true").csv(str(path))


def _safe_ratio(num: F.Column, den: F.Column) -> F.Column:
    # per requirements: denominator=0 -> 0
    return F.when(den == 0, F.lit(0.0)).otherwise(num / den)


def _write_partitioned(df: DataFrame, out_path: Path, repartition_n: Optional[int] = None) -> None:
    out = df
    if repartition_n is not None:
        out = out.repartition(int(repartition_n))
    out.write.mode("overwrite").partitionBy("dt").parquet(str(out_path))


def _dq_pk_rate(df: DataFrame, key_cols: list[str], name: str) -> None:
    total = df.count()
    distinct = df.select(*key_cols).distinct().count() if total else 0
    rate = (distinct / total) if total else 1.0
    print(f"[DQ] {name} PK unique rate: {distinct}/{total} = {rate:.2%}")


def _dq_logic_counts(df: DataFrame, name: str) -> None:
    # clicks <= impressions; conversions <= clicks; cost>=0; gmv>=0
    bad_clicks = df.where(F.col("clicks") > F.col("impressions")).count() if "clicks" in df.columns else 0
    bad_convs = df.where(F.col("conversions") > F.col("clicks")).count() if {"conversions", "clicks"}.issubset(df.columns) else 0
    bad_cost = df.where(F.col("cost") < 0).count() if "cost" in df.columns else 0
    bad_gmv = df.where(F.col("gmv_amount") < 0).count() if "gmv_amount" in df.columns else 0
    print(f"[DQ] {name} logic: clicks>impressions rows={bad_clicks}, convs>clicks rows={bad_convs}, cost<0 rows={bad_cost}, gmv<0 rows={bad_gmv}")


def _dq_metric_sanity(df: DataFrame, name: str) -> None:
    total = df.count()
    if total == 0:
        print(f"[DQ] {name} sanity: empty table")
        return

    def _ratio_in_range(col: str, lo: float, hi: float) -> str:
        ok = df.where((F.col(col) >= F.lit(lo)) & (F.col(col) <= F.lit(hi))).count()
        return f"{ok}/{total} = {(ok/total):.2%}"

    def _ratio_ge(col: str, lo: float) -> str:
        ok = df.where(F.col(col) >= F.lit(lo)).count()
        return f"{ok}/{total} = {(ok/total):.2%}"

    if "ctr" in df.columns:
        print(f"[DQ] {name} ctr in [0,1]: {_ratio_in_range('ctr', 0.0, 1.0)}")
    if "cvr" in df.columns:
        print(f"[DQ] {name} cvr in [0,1]: {_ratio_in_range('cvr', 0.0, 1.0)}")
    if "roi" in df.columns:
        print(f"[DQ] {name} roi >= 0: {_ratio_ge('roi', 0.0)}")


def _read_campaign_cost_for_dt(spark, paths: Paths, dt: str) -> DataFrame:
    """
    Provide campaign-day cost for dt.

    Priority:
    1) warehouse/dwd/dwd_campaign_day_cost if exists
    2) data/ods/dt=YYYY-MM-DD/ods_ad_cost.csv if exists
    3) data/ods/ods_ad_cost.csv fallback filtered by dt
    """
    dwd_cost_path = paths.warehouse_table("dwd/dwd_campaign_day_cost")
    if dwd_cost_path.exists():
        return _read_parquet(spark, dwd_cost_path).where(F.col("dt") == dt).select(
            "dt", "campaign_id", F.col("cost").cast("double").alias("cost")
        )

    dt_dir = paths.ods_dt_dir(dt)
    if dt_dir.exists() and (dt_dir / "ods_ad_cost.csv").exists():
        cost = _read_csv(spark, dt_dir / "ods_ad_cost.csv")
        return cost.select("dt", "campaign_id", F.col("cost").cast("double").alias("cost")).where(F.col("dt") == dt)

    cost = _read_csv(spark, paths.ods_dir / "ods_ad_cost.csv")
    return cost.select("dt", "campaign_id", F.col("cost").cast("double").alias("cost")).where(F.col("dt") == dt)


def _campaign_advertiser_map(impression_v2: DataFrame, click_v2: DataFrame, dt: str) -> DataFrame:
    """
    Mapping (dt,campaign_id)->advertiser_id from facts, for cost allocation.
    Prefer impression_v2, fallback to click_v2.
    """
    imp_map = (
        impression_v2.where(F.col("dt") == dt)
        .select("dt", "campaign_id", "advertiser_id")
        .where(F.col("campaign_id").isNotNull() & F.col("advertiser_id").isNotNull())
        .dropDuplicates(["dt", "campaign_id", "advertiser_id"])
    )
    if imp_map.take(1):
        return imp_map
    return (
        click_v2.where(F.col("dt") == dt)
        .select("dt", "campaign_id", "advertiser_id")
        .where(F.col("campaign_id").isNotNull() & F.col("advertiser_id").isNotNull())
        .dropDuplicates(["dt", "campaign_id", "advertiser_id"])
    )


def _tag_dict_rows() -> list[dict]:
    # Tag dictionary is a "system configuration table".
    # Keep it deterministic and stable across runs.
    return [
        {
            "tag_id": "tag_0001",
            "tag_name": "high_active_7d",
            "tag_type": "behavior",
            "rule_desc": "近7天活跃天数>=3（当天有曝光或点击计为活跃）",
            "update_freq": "daily",
        },
        {
            "tag_id": "tag_0002",
            "tag_name": "high_ctr_7d",
            "tag_type": "behavior",
            "rule_desc": "近7天 CTR>=0.04 且 impressions>=50",
            "update_freq": "daily",
        },
        {
            "tag_id": "tag_0003",
            "tag_name": "high_cvr_7d",
            "tag_type": "behavior",
            "rule_desc": "近7天 CVR>=0.06 且 clicks>=10",
            "update_freq": "daily",
        },
        {
            "tag_id": "tag_0004",
            "tag_name": "high_value_7d",
            "tag_type": "value",
            "rule_desc": "近7天 GMV>=300",
            "update_freq": "daily",
        },
        {
            "tag_id": "tag_0005",
            "tag_name": "ad_clicker",
            "tag_type": "behavior",
            "rule_desc": "当天 clicks>=3",
            "update_freq": "daily",
        },
        {
            "tag_id": "tag_0006",
            "tag_name": "ad_viewer",
            "tag_type": "behavior",
            "rule_desc": "当天 impressions>=20",
            "update_freq": "daily",
        },
        {
            "tag_id": "tag_0007",
            "tag_name": "converter",
            "tag_type": "behavior",
            "rule_desc": "当天 conversions>=1",
            "update_freq": "daily",
        },
        {
            "tag_id": "tag_0008",
            "tag_name": "bounce_user",
            "tag_type": "risk",
            "rule_desc": "当天 impressions>0 且 clicks=0 且 conversions=0（只看不点不转）",
            "update_freq": "daily",
        },
    ]


def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _build_tag_dict_sql() -> str:
    rows = _tag_dict_rows()
    values = ",\n  ".join(
        f"({_sql_quote(r['tag_id'])},{_sql_quote(r['tag_name'])},{_sql_quote(r['tag_type'])},{_sql_quote(r['rule_desc'])},{_sql_quote(r['update_freq'])})"
        for r in rows
    )
    return f"""
WITH v(tag_id, tag_name, tag_type, rule_desc, update_freq) AS (
  VALUES
  {values}
)
SELECT
  tag_id,
  tag_name,
  tag_type,
  rule_desc,
  update_freq,
  current_timestamp() AS create_time
FROM v
""".strip()


def _dt_range_7d(dt: str) -> List[str]:
    d = datetime.strptime(dt, "%Y-%m-%d")
    return [(d - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]


def _dq_rate_in_range(df: DataFrame, col: str, lo: float, hi: float, name: str) -> None:
    total = df.count()
    if total == 0:
        print(f"[DQ] {name} {col} in [{lo},{hi}]: empty table")
        return
    ok = df.where((F.col(col) >= F.lit(lo)) & (F.col(col) <= F.lit(hi))).count()
    print(f"[DQ] {name} {col} in [{lo},{hi}]: {ok}/{total} = {(ok/total):.2%}")


def _dq_rate_ge(df: DataFrame, col: str, lo: float, name: str) -> None:
    total = df.count()
    if total == 0:
        print(f"[DQ] {name} {col}>={lo}: empty table")
        return
    ok = df.where(F.col(col) >= F.lit(lo)).count()
    print(f"[DQ] {name} {col}>={lo}: {ok}/{total} = {(ok/total):.2%}")


def _build_tags(
    spark,
    paths: Paths,
    dt: str,
    dws_user_daily_today: DataFrame,
    dws_user_daily_root: Path,
) -> None:
    """
    DWS Tagging module:
    - dws_tag_dict: non-partition tag dictionary
    - dws_user_tag_snapshot: dt-partition snapshot (one user per row)
    - dws_tag_quality_daily: dt-partition tag quality metrics
    """
    # 1) tag dict (non-partition)
    # Build tag_dict with pure SQL to avoid Python worker dependency/version mismatch.
    tag_dict = spark.sql(_build_tag_dict_sql())
    out_tag_dict = paths.warehouse_table("dws/dws_tag_dict")
    tag_dict.coalesce(1).write.mode("overwrite").parquet(str(out_tag_dict))

    # 2) user tag snapshot (dt partition)
    # day features: from dws_user_daily(dt)
    day = dws_user_daily_today.select(
        "dt",
        "user_id",
        F.col("imp_cnt").cast("long").alias("feature_day_impressions"),
        F.col("clk_cnt").cast("long").alias("feature_day_clicks"),
        F.col("conv_cnt").cast("long").alias("feature_day_conversions"),
        F.col("gmv_amount").cast("double").alias("feature_day_gmv"),
        F.col("last_event_time").alias("day_last_event_time"),
    )

    # 7d features: dt-6 .. dt aggregated from dws_user_daily
    window_dts = _dt_range_7d(dt)
    if dws_user_daily_root.exists():
        hist = spark.read.parquet(str(dws_user_daily_root)).where(F.col("dt").isin(window_dts))
    else:
        hist = dws_user_daily_today.where(F.col("dt").isin(window_dts))

    hist = hist.select(
        "dt",
        "user_id",
        F.col("imp_cnt").cast("long").alias("imp_cnt"),
        F.col("clk_cnt").cast("long").alias("clk_cnt"),
        F.col("conv_cnt").cast("long").alias("conv_cnt"),
        F.col("gmv_amount").cast("double").alias("gmv_amount"),
        F.col("last_event_time").alias("last_event_time"),
    )

    hist_agg = (
        hist.withColumn("active_flag", F.when((F.col("imp_cnt") > 0) | (F.col("clk_cnt") > 0), F.lit(1)).otherwise(F.lit(0)))
        .groupBy("user_id")
        .agg(
            F.sum("imp_cnt").alias("feature_7d_impressions"),
            F.sum("clk_cnt").alias("feature_7d_clicks"),
            F.sum("conv_cnt").alias("feature_7d_conversions"),
            F.sum(F.coalesce(F.col("gmv_amount"), F.lit(0.0))).alias("feature_7d_gmv"),
            F.sum("active_flag").alias("feature_7d_active_days"),
            F.max("last_event_time").alias("last_event_time"),
        )
        .withColumn("feature_7d_ctr", _safe_ratio(F.col("feature_7d_clicks").cast("double"), F.col("feature_7d_impressions").cast("double")))
        .withColumn("feature_7d_cvr", _safe_ratio(F.col("feature_7d_conversions").cast("double"), F.col("feature_7d_clicks").cast("double")))
        .select(
            "user_id",
            "feature_7d_impressions",
            "feature_7d_clicks",
            "feature_7d_conversions",
            "feature_7d_gmv",
            "feature_7d_ctr",
            "feature_7d_cvr",
            "feature_7d_active_days",
            "last_event_time",
        )
    )

    base = (
        day.join(hist_agg, on="user_id", how="left")
        .withColumn("dt", F.lit(dt))
        .na.fill(
            {
                "feature_7d_impressions": 0,
                "feature_7d_clicks": 0,
                "feature_7d_conversions": 0,
                "feature_7d_gmv": 0.0,
                "feature_7d_ctr": 0.0,
                "feature_7d_cvr": 0.0,
                "feature_7d_active_days": 0,
                "feature_day_impressions": 0,
                "feature_day_clicks": 0,
                "feature_day_conversions": 0,
                "feature_day_gmv": 0.0,
            }
        )
        .withColumn("last_event_time", F.coalesce(F.col("last_event_time"), F.col("day_last_event_time")))
        .drop("day_last_event_time")
    )

    # Tag rules
    cond_high_active_7d = F.col("feature_7d_active_days") >= F.lit(3)
    cond_high_ctr_7d = (F.col("feature_7d_ctr") >= F.lit(0.04)) & (F.col("feature_7d_impressions") >= F.lit(50))
    cond_high_cvr_7d = (F.col("feature_7d_cvr") >= F.lit(0.06)) & (F.col("feature_7d_clicks") >= F.lit(10))
    cond_high_value_7d = F.col("feature_7d_gmv") >= F.lit(300.0)
    cond_ad_clicker = F.col("feature_day_clicks") >= F.lit(3)
    cond_ad_viewer = F.col("feature_day_impressions") >= F.lit(20)
    cond_converter = F.col("feature_day_conversions") >= F.lit(1)
    cond_bounce = (F.col("feature_day_impressions") > F.lit(0)) & (F.col("feature_day_clicks") == F.lit(0)) & (F.col("feature_day_conversions") == F.lit(0))

    tags_tmp = F.array(
        F.when(cond_high_active_7d, F.lit("high_active_7d")),
        F.when(cond_high_ctr_7d, F.lit("high_ctr_7d")),
        F.when(cond_high_cvr_7d, F.lit("high_cvr_7d")),
        F.when(cond_high_value_7d, F.lit("high_value_7d")),
        F.when(cond_ad_clicker, F.lit("ad_clicker")),
        F.when(cond_ad_viewer, F.lit("ad_viewer")),
        F.when(cond_converter, F.lit("converter")),
        F.when(cond_bounce, F.lit("bounce_user")),
    )

    snapshot = (
        base.withColumn("tags_tmp", tags_tmp)
        .withColumn("tags", F.expr("filter(tags_tmp, x -> x is not null)"))
        .drop("tags_tmp")
        .withColumn("tag_cnt", F.size(F.col("tags")).cast("int"))
        .select(
            "dt",
            "user_id",
            "tags",
            "tag_cnt",
            "feature_7d_impressions",
            "feature_7d_clicks",
            "feature_7d_conversions",
            "feature_7d_gmv",
            "feature_7d_ctr",
            "feature_7d_cvr",
            "feature_day_impressions",
            "feature_day_clicks",
            "feature_day_conversions",
            "feature_day_gmv",
            "last_event_time",
        )
    )

    out_snapshot = paths.warehouse_table("dws/dws_user_tag_snapshot")
    _write_partitioned(snapshot, out_snapshot, repartition_n=8)

    # DQ for snapshot
    _dq_pk_rate(snapshot, ["dt", "user_id"], "dws_user_tag_snapshot")
    _dq_rate_ge(snapshot, "tag_cnt", 0, "dws_user_tag_snapshot")
    _dq_rate_in_range(snapshot, "feature_7d_ctr", 0.0, 1.0, "dws_user_tag_snapshot")
    _dq_rate_in_range(snapshot, "feature_7d_cvr", 0.0, 1.0, "dws_user_tag_snapshot")

    # 3) tag quality daily
    total_user_cnt = snapshot.select("user_id").distinct().count()

    exploded = snapshot.select(
        "dt",
        "user_id",
        "feature_7d_ctr",
        "feature_7d_cvr",
        "feature_7d_gmv",
        F.explode_outer("tags").alias("tag_name"),
    ).where(F.col("tag_name").isNotNull())

    agg = (
        exploded.groupBy("dt", "tag_name")
        .agg(
            F.countDistinct("user_id").alias("tagged_user_cnt"),
            F.avg("feature_7d_ctr").alias("avg_ctr_tagged"),
            F.avg("feature_7d_cvr").alias("avg_cvr_tagged"),
            F.avg("feature_7d_gmv").alias("avg_gmv_tagged"),
        )
    )

    # Ensure all tags appear even if tagged_user_cnt=0
    all_tags = tag_dict.select("tag_name").distinct().withColumn("dt", F.lit(dt))
    quality = (
        all_tags.join(agg, on=["dt", "tag_name"], how="left")
        .withColumn("total_user_cnt", F.lit(int(total_user_cnt)))
        .na.fill({"tagged_user_cnt": 0, "avg_ctr_tagged": 0.0, "avg_cvr_tagged": 0.0, "avg_gmv_tagged": 0.0})
        .withColumn("coverage_rate", _safe_ratio(F.col("tagged_user_cnt").cast("double"), F.col("total_user_cnt").cast("double")))
        .select(
            "dt",
            "tag_name",
            F.col("tagged_user_cnt").cast("long").alias("tagged_user_cnt"),
            F.col("total_user_cnt").cast("long").alias("total_user_cnt"),
            "coverage_rate",
            F.col("avg_ctr_tagged").cast("double").alias("avg_ctr_tagged"),
            F.col("avg_cvr_tagged").cast("double").alias("avg_cvr_tagged"),
            F.col("avg_gmv_tagged").cast("double").alias("avg_gmv_tagged"),
        )
    )

    out_quality = paths.warehouse_table("dws/dws_tag_quality_daily")
    _write_partitioned(quality, out_quality, repartition_n=1)

    # DQ for quality
    _dq_pk_rate(quality, ["dt", "tag_name"], "dws_tag_quality_daily")
    _dq_rate_in_range(quality, "coverage_rate", 0.0, 1.0, "dws_tag_quality_daily")
    bad_cnt = quality.where(F.col("tagged_user_cnt") > F.col("total_user_cnt")).count()
    print(f"[DQ] dws_tag_quality_daily tagged_user_cnt<=total_user_cnt violated rows={bad_cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build light DWS aggregates from DWD v2.")
    parser.add_argument("--dt", required=True, help="YYYY-MM-DD (single-day incremental run)")
    args = parser.parse_args()

    dt = args.dt
    paths = Paths(project_root=_project_root())

    # Ensure driver/executor Python minor version一致，避免 Python worker 版本不匹配报错。
    spark = build_spark(
        app_name=f"warehouse_02_build_dws_dt={dt}",
        extra_conf={
            "spark.pyspark.driver.python": sys.executable,
            "spark.pyspark.python": sys.executable,
        },
    )

    # -------------------- Inputs (DWD v2) --------------------
    imp_path = paths.warehouse_table("dwd/dwd_ad_impression_detail_v2")
    clk_path = paths.warehouse_table("dwd/dwd_ad_click_detail_v2")
    cvr_path = paths.warehouse_table("dwd/dwd_ad_conversion_detail_v2")

    imp = _read_parquet(spark, imp_path).where(F.col("dt") == dt)
    clk = _read_parquet(spark, clk_path).where(F.col("dt") == dt)
    cvr = _read_parquet(spark, cvr_path).where(F.col("dt") == dt)

    # Cost (campaign-day)
    campaign_cost = _read_campaign_cost_for_dt(spark, paths, dt)

    # -------------------- 1) dws_campaign_daily --------------------
    imp_campaign = imp.groupBy("dt", "campaign_id").agg(F.count(F.lit(1)).alias("impressions"))
    clk_campaign = clk.groupBy("dt", "campaign_id").agg(F.count(F.lit(1)).alias("clicks"))
    cvr_campaign = cvr.groupBy("dt", "campaign_id").agg(
        F.count(F.lit(1)).alias("conversions"),
        F.sum(F.coalesce(F.col("gmv_amount").cast("double"), F.lit(0.0))).alias("gmv_amount"),
    )

    dws_campaign_daily = (
        imp_campaign.join(clk_campaign, on=["dt", "campaign_id"], how="full")
        .join(cvr_campaign, on=["dt", "campaign_id"], how="full")
        .join(campaign_cost, on=["dt", "campaign_id"], how="left")
        .na.fill({"impressions": 0, "clicks": 0, "conversions": 0, "gmv_amount": 0.0, "cost": 0.0})
        .withColumn("ctr", _safe_ratio(F.col("clicks").cast("double"), F.col("impressions").cast("double")))
        .withColumn("cvr", _safe_ratio(F.col("conversions").cast("double"), F.col("clicks").cast("double")))
        .withColumn("rpm", _safe_ratio(F.col("gmv_amount").cast("double") * F.lit(1000.0), F.col("impressions").cast("double")))
        .withColumn("roi", _safe_ratio(F.col("gmv_amount").cast("double"), F.col("cost").cast("double")))
        .select(
            "dt",
            "campaign_id",
            F.col("impressions").cast("long").alias("impressions"),
            F.col("clicks").cast("long").alias("clicks"),
            F.col("conversions").cast("long").alias("conversions"),
            F.col("gmv_amount").cast("double").alias("gmv_amount"),
            F.col("cost").cast("double").alias("cost"),
            "ctr",
            "cvr",
            "rpm",
            "roi",
        )
    )

    out_campaign = paths.warehouse_table("dws/dws_campaign_daily")
    _write_partitioned(dws_campaign_daily, out_campaign, repartition_n=1)

    _dq_pk_rate(dws_campaign_daily, ["dt", "campaign_id"], "dws_campaign_daily")
    _dq_logic_counts(dws_campaign_daily, "dws_campaign_daily")
    _dq_metric_sanity(dws_campaign_daily, "dws_campaign_daily")

    # -------------------- 2) dws_advertiser_daily --------------------
    # Facts already include advertiser_id (from DWD v2 join)
    imp_adv = imp.where(F.col("advertiser_id").isNotNull()).groupBy("dt", "advertiser_id").agg(F.count(F.lit(1)).alias("impressions"))
    clk_adv = clk.where(F.col("advertiser_id").isNotNull()).groupBy("dt", "advertiser_id").agg(F.count(F.lit(1)).alias("clicks"))
    cvr_adv = cvr.where(F.col("advertiser_id").isNotNull()).groupBy("dt", "advertiser_id").agg(
        F.count(F.lit(1)).alias("conversions"),
        F.sum(F.coalesce(F.col("gmv_amount").cast("double"), F.lit(0.0))).alias("gmv_amount"),
    )

    # Cost allocation:
    # - build mapping (dt,campaign_id)->advertiser_id distinct from facts
    # - if a campaign maps to multiple advertisers, split cost equally among advertisers for that campaign
    camp_adv_map = _campaign_advertiser_map(imp, clk, dt)
    adv_cnt = camp_adv_map.groupBy("dt", "campaign_id").agg(F.countDistinct("advertiser_id").alias("adv_cnt"))
    adv_cost = (
        campaign_cost.join(adv_cnt, on=["dt", "campaign_id"], how="left")
        .join(camp_adv_map, on=["dt", "campaign_id"], how="left")
        .withColumn("adv_cnt", F.when(F.col("adv_cnt").isNull() | (F.col("adv_cnt") == 0), F.lit(1)).otherwise(F.col("adv_cnt")))
        .withColumn("cost_part", F.col("cost") / F.col("adv_cnt"))
        .groupBy("dt", "advertiser_id")
        .agg(F.sum(F.coalesce(F.col("cost_part"), F.lit(0.0))).alias("cost"))
    )

    dws_advertiser_daily = (
        imp_adv.join(clk_adv, on=["dt", "advertiser_id"], how="full")
        .join(cvr_adv, on=["dt", "advertiser_id"], how="full")
        .join(adv_cost, on=["dt", "advertiser_id"], how="left")
        .na.fill({"impressions": 0, "clicks": 0, "conversions": 0, "gmv_amount": 0.0, "cost": 0.0})
        .withColumn("ctr", _safe_ratio(F.col("clicks").cast("double"), F.col("impressions").cast("double")))
        .withColumn("cvr", _safe_ratio(F.col("conversions").cast("double"), F.col("clicks").cast("double")))
        .withColumn("rpm", _safe_ratio(F.col("gmv_amount").cast("double") * F.lit(1000.0), F.col("impressions").cast("double")))
        .withColumn("roi", _safe_ratio(F.col("gmv_amount").cast("double"), F.col("cost").cast("double")))
        .select(
            "dt",
            "advertiser_id",
            F.col("impressions").cast("long").alias("impressions"),
            F.col("clicks").cast("long").alias("clicks"),
            F.col("conversions").cast("long").alias("conversions"),
            F.col("gmv_amount").cast("double").alias("gmv_amount"),
            F.col("cost").cast("double").alias("cost"),
            "ctr",
            "cvr",
            "rpm",
            "roi",
        )
    )

    out_adv = paths.warehouse_table("dws/dws_advertiser_daily")
    _write_partitioned(dws_advertiser_daily, out_adv, repartition_n=1)

    _dq_pk_rate(dws_advertiser_daily, ["dt", "advertiser_id"], "dws_advertiser_daily")
    _dq_logic_counts(dws_advertiser_daily, "dws_advertiser_daily")
    _dq_metric_sanity(dws_advertiser_daily, "dws_advertiser_daily")

    # -------------------- 3) dws_user_daily --------------------
    imp_user = imp.groupBy("dt", "user_id").agg(
        F.count(F.lit(1)).alias("imp_cnt"),
        F.countDistinct("campaign_id").alias("imp_campaign_cnt"),
        F.max(F.col("event_time")).alias("last_imp_time"),
    )
    clk_user = clk.groupBy("dt", "user_id").agg(
        F.count(F.lit(1)).alias("clk_cnt"),
        F.countDistinct("campaign_id").alias("clk_campaign_cnt"),
        F.max(F.col("event_time")).alias("last_clk_time"),
    )
    cvr_user = cvr.groupBy("dt", "user_id").agg(
        F.count(F.lit(1)).alias("conv_cnt"),
        F.sum(F.coalesce(F.col("gmv_amount").cast("double"), F.lit(0.0))).alias("gmv_amount"),
        F.max(F.col("conv_time")).alias("last_conv_time"),
    )

    dws_user_daily = (
        imp_user.join(clk_user, on=["dt", "user_id"], how="full")
        .join(cvr_user, on=["dt", "user_id"], how="full")
        .na.fill({"imp_cnt": 0, "clk_cnt": 0, "conv_cnt": 0, "gmv_amount": 0.0, "imp_campaign_cnt": 0, "clk_campaign_cnt": 0})
        .withColumn("active_campaign_cnt", F.greatest(F.col("imp_campaign_cnt").cast("long"), F.col("clk_campaign_cnt").cast("long")))
        .withColumn("last_event_time", F.greatest(F.col("last_imp_time"), F.col("last_clk_time"), F.col("last_conv_time")))
        .withColumn("ctr", _safe_ratio(F.col("clk_cnt").cast("double"), F.col("imp_cnt").cast("double")))
        .withColumn("cvr", _safe_ratio(F.col("conv_cnt").cast("double"), F.col("clk_cnt").cast("double")))
        .select(
            "dt",
            "user_id",
            F.col("imp_cnt").cast("long").alias("imp_cnt"),
            F.col("clk_cnt").cast("long").alias("clk_cnt"),
            F.col("conv_cnt").cast("long").alias("conv_cnt"),
            F.col("gmv_amount").cast("double").alias("gmv_amount"),
            F.col("active_campaign_cnt").cast("long").alias("active_campaign_cnt"),
            "last_event_time",
            "ctr",
            "cvr",
        )
    )

    out_user = paths.warehouse_table("dws/dws_user_daily")
    _write_partitioned(dws_user_daily, out_user, repartition_n=8)

    _dq_pk_rate(dws_user_daily, ["dt", "user_id"], "dws_user_daily")
    # user_daily has different metric names; do minimal sanity checks aligned to its schema
    bad_ctr = dws_user_daily.where((F.col("ctr") < 0) | (F.col("ctr") > 1)).count()
    bad_cvr = dws_user_daily.where((F.col("cvr") < 0) | (F.col("cvr") > 1)).count()
    bad_gmv = dws_user_daily.where(F.col("gmv_amount") < 0).count()
    print(f"[DQ] dws_user_daily sanity: ctr out-of-range rows={bad_ctr}, cvr out-of-range rows={bad_cvr}, gmv<0 rows={bad_gmv}")

    # -------------------- Tagging module (DWS) --------------------
    _build_tags(
        spark=spark,
        paths=paths,
        dt=dt,
        dws_user_daily_today=dws_user_daily,
        dws_user_daily_root=out_user,
    )

    spark.stop()


if __name__ == "__main__":
    main()

