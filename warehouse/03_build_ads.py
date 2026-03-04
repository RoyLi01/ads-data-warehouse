from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark.common.spark_session import build_spark


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def warehouse_dir(self) -> Path:
        return self.project_root / "warehouse"

    def warehouse_table(self, rel: str) -> Path:
        return self.warehouse_dir / rel


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_parquet(spark, path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet path: {path}")
    return spark.read.parquet(str(path))


def _safe_ratio(num: F.Column, den: F.Column, null_if_den_zero: bool = False) -> F.Column:
    """
    Safe division.
    - if null_if_den_zero=True: denominator=0 -> NULL
    - else: denominator=0 -> 0
    """
    if null_if_den_zero:
        return F.when(den == 0, F.lit(None)).otherwise(num / den)
    return F.when(den == 0, F.lit(0.0)).otherwise(num / den)


def _write_partitioned(df: DataFrame, out_path: Path, repartition_n: int | None = None) -> None:
    out = df
    if repartition_n is not None:
        out = out.repartition(int(repartition_n))
    out.write.mode("overwrite").partitionBy("dt").parquet(str(out_path))


def _dq_pk_rate(df: DataFrame, key_cols: list[str], name: str) -> None:
    total = df.count()
    distinct = df.select(*key_cols).distinct().count() if total else 0
    rate = (distinct / total) if total else 1.0
    print(f"[DQ] {name} PK unique rate: {distinct}/{total} = {rate:.2%}")


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


def _prev_dt(dt: str) -> str:
    d = datetime.strptime(dt, "%Y-%m-%d")
    return (d - timedelta(days=1)).strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ADS layer reports from DWS (incremental by dt).")
    parser.add_argument("--dt", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    dt = args.dt
    paths = Paths(project_root=_project_root())

    spark = build_spark(app_name=f"warehouse_03_build_ads_dt={dt}")

    # -------------------- Inputs (DWS) --------------------
    dws_campaign_daily = _read_parquet(spark, paths.warehouse_table("dws/dws_campaign_daily")).where(F.col("dt") == dt)
    dws_advertiser_daily = _read_parquet(spark, paths.warehouse_table("dws/dws_advertiser_daily")).where(F.col("dt") == dt)
    dws_user_daily = _read_parquet(spark, paths.warehouse_table("dws/dws_user_daily")).where(F.col("dt") == dt)
    dws_user_tag_snapshot = _read_parquet(spark, paths.warehouse_table("dws/dws_user_tag_snapshot")).where(F.col("dt") == dt)
    dws_tag_quality_daily = _read_parquet(spark, paths.warehouse_table("dws/dws_tag_quality_daily")).where(F.col("dt") == dt)

    # -------------------- 1) ads_kpi_overview_daily --------------------
    # Grain: 1 row per dt
    overview = (
        dws_campaign_daily.groupBy("dt")
        .agg(
            F.sum(F.col("impressions").cast("double")).alias("impressions"),
            F.sum(F.col("clicks").cast("double")).alias("clicks"),
            F.sum(F.col("conversions").cast("double")).alias("conversions"),
            F.sum(F.col("gmv_amount").cast("double")).alias("gmv"),
            F.sum(F.col("cost").cast("double")).alias("cost"),
        )
        .join(dws_user_daily.groupBy("dt").agg(F.count(F.lit(1)).alias("active_users")), on="dt", how="left")
        .na.fill({"active_users": 0})
        .withColumn("ctr", _safe_ratio(F.col("clicks"), F.col("impressions")))
        .withColumn("cvr", _safe_ratio(F.col("conversions"), F.col("clicks")))
        .withColumn("rpm", _safe_ratio(F.col("gmv") * F.lit(1000.0), F.col("impressions")))
        .withColumn("arpu", _safe_ratio(F.col("gmv"), F.col("active_users").cast("double")))
        .withColumn("roi", _safe_ratio(F.col("gmv"), F.col("cost")))
        .select(
            "dt",
            F.col("impressions").cast("long").alias("impressions"),
            F.col("clicks").cast("long").alias("clicks"),
            F.col("conversions").cast("long").alias("conversions"),
            F.col("gmv").cast("double").alias("gmv"),
            F.col("cost").cast("double").alias("cost"),
            "ctr",
            "cvr",
            "rpm",
            "arpu",
            F.col("active_users").cast("long").alias("active_users"),
            "roi",
        )
    )

    out_overview = paths.warehouse_table("ads/ads_kpi_overview_daily")
    _write_partitioned(overview, out_overview, repartition_n=1)

    # DQ: should be 1 row for dt
    overview_cnt = overview.count()
    print(f"[DQ] ads_kpi_overview_daily row count for dt={dt}: {overview_cnt}")
    _dq_rate_in_range(overview, "ctr", 0.0, 1.0, "ads_kpi_overview_daily")
    _dq_rate_in_range(overview, "cvr", 0.0, 1.0, "ads_kpi_overview_daily")
    _dq_rate_ge(overview, "roi", 0.0, "ads_kpi_overview_daily")

    # -------------------- 2) ads_campaign_ranking_daily --------------------
    # Grain: dt + campaign_id
    # Policy: keep impressions=0 campaigns; they naturally rank lower.
    campaign_base = dws_campaign_daily.select(
        "dt",
        "campaign_id",
        "impressions",
        "clicks",
        "conversions",
        F.col("gmv_amount").alias("gmv"),
        "cost",
        "ctr",
        "cvr",
        "rpm",
        "roi",
    )

    w_gmv = Window.partitionBy("dt").orderBy(F.col("gmv").desc(), F.col("campaign_id").asc())
    w_roi = Window.partitionBy("dt").orderBy(F.col("roi").desc(), F.col("campaign_id").asc())

    campaign_ranking = (
        campaign_base.withColumn("rank_by_gmv", F.row_number().over(w_gmv))
        .withColumn("rank_by_roi", F.row_number().over(w_roi))
        .select(
            "dt",
            "campaign_id",
            F.col("impressions").cast("long").alias("impressions"),
            F.col("clicks").cast("long").alias("clicks"),
            F.col("conversions").cast("long").alias("conversions"),
            F.col("gmv").cast("double").alias("gmv"),
            F.col("cost").cast("double").alias("cost"),
            F.col("ctr").cast("double").alias("ctr"),
            F.col("cvr").cast("double").alias("cvr"),
            F.col("rpm").cast("double").alias("rpm"),
            F.col("roi").cast("double").alias("roi"),
            F.col("rank_by_gmv").cast("int").alias("rank_by_gmv"),
            F.col("rank_by_roi").cast("int").alias("rank_by_roi"),
        )
    )

    out_campaign_ranking = paths.warehouse_table("ads/ads_campaign_ranking_daily")
    _write_partitioned(campaign_ranking, out_campaign_ranking, repartition_n=1)

    _dq_pk_rate(campaign_ranking, ["dt", "campaign_id"], "ads_campaign_ranking_daily")

    # -------------------- 3) ads_advertiser_dashboard_daily --------------------
    # Grain: dt + advertiser_id
    # revenue_growth_rate = (gmv - prev_gmv) / prev_gmv, prev=0 -> NULL
    prev = _prev_dt(dt)
    prev_part = paths.warehouse_table("dws/dws_advertiser_daily") / f"dt={prev}"
    prev_gmv_df: DataFrame | None = None
    if prev_part.exists():
        prev_gmv_df = (
            _read_parquet(spark, paths.warehouse_table("dws/dws_advertiser_daily"))
            .where(F.col("dt") == prev)
            .select(F.col("advertiser_id"), F.col("gmv_amount").cast("double").alias("prev_gmv"))
        )

    dash = dws_advertiser_daily.select(
        "dt",
        "advertiser_id",
        "impressions",
        "clicks",
        "conversions",
        F.col("gmv_amount").alias("gmv"),
        "cost",
        "ctr",
        "cvr",
        "roi",
    )
    if prev_gmv_df is not None:
        dash = dash.join(prev_gmv_df, on="advertiser_id", how="left")
    else:
        dash = dash.withColumn("prev_gmv", F.lit(None).cast("double"))

    dash = (
        dash.withColumn(
            "revenue_growth_rate",
            F.when(F.col("prev_gmv").isNull() | (F.col("prev_gmv") == 0), F.lit(None)).otherwise(
                (F.col("gmv").cast("double") - F.col("prev_gmv")) / F.col("prev_gmv")
            ),
        )
        .withColumn("retention_flag", F.when(F.col("cost").cast("double") > 0, F.lit(1)).otherwise(F.lit(0)))
        .select(
            "dt",
            "advertiser_id",
            F.col("impressions").cast("long").alias("impressions"),
            F.col("clicks").cast("long").alias("clicks"),
            F.col("conversions").cast("long").alias("conversions"),
            F.col("gmv").cast("double").alias("gmv"),
            F.col("cost").cast("double").alias("cost"),
            F.col("ctr").cast("double").alias("ctr"),
            F.col("cvr").cast("double").alias("cvr"),
            F.col("roi").cast("double").alias("roi"),
            "revenue_growth_rate",
            F.col("retention_flag").cast("int").alias("retention_flag"),
        )
    )

    out_dash = paths.warehouse_table("ads/ads_advertiser_dashboard_daily")
    _write_partitioned(dash, out_dash, repartition_n=1)

    _dq_pk_rate(dash, ["dt", "advertiser_id"], "ads_advertiser_dashboard_daily")
    # growth_rate: allow null, but not inf
    inf_rows = dash.where(F.col("revenue_growth_rate").isin(float("inf"), float("-inf"))).count()
    print(f"[DQ] ads_advertiser_dashboard_daily growth_rate inf rows={inf_rows}")

    # -------------------- 4) ads_tag_effectiveness_daily --------------------
    # Grain: dt + tag_name
    # Policy: copy from dws_tag_quality_daily
    tag_eff = dws_tag_quality_daily.select(
        "dt",
        "tag_name",
        "coverage_rate",
        "tagged_user_cnt",
        "avg_ctr_tagged",
        "avg_cvr_tagged",
        "avg_gmv_tagged",
    )
    out_tag_eff = paths.warehouse_table("ads/ads_tag_effectiveness_daily")
    _write_partitioned(tag_eff, out_tag_eff, repartition_n=1)

    _dq_pk_rate(tag_eff, ["dt", "tag_name"], "ads_tag_effectiveness_daily")
    _dq_rate_in_range(tag_eff, "coverage_rate", 0.0, 1.0, "ads_tag_effectiveness_daily")

    spark.stop()


if __name__ == "__main__":
    main()

