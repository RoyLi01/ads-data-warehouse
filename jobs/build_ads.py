from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import sys

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from common.spark_session import build_spark


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def warehouse_dir(self) -> Path:
        return self.project_root / "warehouse"

    def warehouse_table(self, rel: str) -> Path:
        return self.warehouse_dir / rel


def _project_root() -> Path:
    # jobs/*.py -> jobs -> project root
    return Path(__file__).resolve().parents[1]


def _read_parquet(spark, path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet path: {path}")
    return spark.read.parquet(str(path))


def _table_exists(spark, table_name: str) -> bool:
    try:
        return spark.catalog.tableExists(table_name)
    except Exception:
        return False


def _read_table_or_parquet(spark, table_name: str, fallback_path: Path) -> DataFrame:
    """
    Prefer Hive/Spark catalog tables first.

    Keep parquet fallback so the ADS job remains runnable while preserving the
    current local project layout.
    """
    if _table_exists(spark, table_name):
        return spark.table(table_name)
    return _read_parquet(spark, fallback_path)


def _sql_string(s: str) -> str:
    return s.replace("'", "''")


def _write_partitioned(df: DataFrame, out_path: Path, repartition_n: Optional[int] = None) -> None:
    out = df
    if repartition_n is not None:
        out = out.repartition(int(repartition_n))
    out.write.mode("overwrite").partitionBy("dt").parquet(str(out_path))


def _write_partitioned_table_or_path(
    spark,
    df: DataFrame,
    table_name: str,
    out_path: Path,
    dt: str,
    data_cols: list[str],
    repartition_n: Optional[int] = None,
) -> None:
    """
    Prefer Hive-style table writes when the ADS target table exists.

    Python keeps orchestration and compatibility; report logic itself is modeled
    in Spark SQL so this file aligns with build_dwd.py / build_dws.py.
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

    _write_partitioned(df, out_path, repartition_n=repartition_n)


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
    parser.add_argument("--no_dq", action="store_true", help="Skip DQ prints that trigger expensive counts.")
    args = parser.parse_args()

    dt = args.dt
    prev_dt = _prev_dt(dt)
    paths = Paths(project_root=_project_root())

    spark = build_spark(
        app_name=f"build_ads_dt={dt}",
        extra_conf={
            "spark.pyspark.driver.python": sys.executable,
            "spark.pyspark.python": sys.executable,
        },
    )

    # Python only coordinates table/path fallback. Core ADS reporting is SQL-first below.
    dws_campaign_daily_all = _read_table_or_parquet(spark, "dws.campaign_daily", paths.warehouse_table("dws/dws_campaign_daily"))
    dws_advertiser_daily_all = _read_table_or_parquet(spark, "dws.advertiser_daily", paths.warehouse_table("dws/dws_advertiser_daily"))
    dws_user_daily_all = _read_table_or_parquet(spark, "dws.user_daily", paths.warehouse_table("dws/dws_user_daily"))
    dws_tag_quality_daily_all = _read_table_or_parquet(spark, "dws.tag_quality_daily", paths.warehouse_table("dws/dws_tag_quality_daily"))

    dws_campaign_daily_all.createOrReplaceTempView("dws_campaign_daily_all_v")
    dws_advertiser_daily_all.createOrReplaceTempView("dws_advertiser_daily_all_v")
    dws_user_daily_all.createOrReplaceTempView("dws_user_daily_all_v")
    dws_tag_quality_daily_all.createOrReplaceTempView("dws_tag_quality_daily_all_v")

    spark.sql(
        f"""
CREATE OR REPLACE TEMP VIEW dws_campaign_daily_v AS
SELECT *
FROM dws_campaign_daily_all_v
WHERE dt = '{_sql_string(dt)}'
""".strip()
    )
    spark.sql(
        f"""
CREATE OR REPLACE TEMP VIEW dws_advertiser_daily_v AS
SELECT *
FROM dws_advertiser_daily_all_v
WHERE dt = '{_sql_string(dt)}'
""".strip()
    )
    spark.sql(
        f"""
CREATE OR REPLACE TEMP VIEW dws_user_daily_v AS
SELECT *
FROM dws_user_daily_all_v
WHERE dt = '{_sql_string(dt)}'
""".strip()
    )
    spark.sql(
        f"""
CREATE OR REPLACE TEMP VIEW dws_tag_quality_daily_v AS
SELECT *
FROM dws_tag_quality_daily_all_v
WHERE dt = '{_sql_string(dt)}'
""".strip()
    )
    spark.sql(
        f"""
CREATE OR REPLACE TEMP VIEW dws_advertiser_daily_prev_v AS
SELECT advertiser_id, CAST(gmv_amount AS DOUBLE) AS prev_gmv
FROM dws_advertiser_daily_all_v
WHERE dt = '{_sql_string(prev_dt)}'
""".strip()
    )

    # 1) ads.kpi_overview_daily
    sql_kpi_overview_daily = """
WITH campaign_agg AS (
  SELECT
    dt,
    SUM(CAST(impressions AS DOUBLE)) AS impressions,
    SUM(CAST(clicks AS DOUBLE)) AS clicks,
    SUM(CAST(conversions AS DOUBLE)) AS conversions,
    SUM(CAST(gmv_amount AS DOUBLE)) AS gmv,
    SUM(CAST(cost AS DOUBLE)) AS cost
  FROM dws_campaign_daily_v
  GROUP BY dt
),
active_users_agg AS (
  SELECT
    dt,
    COUNT(1) AS active_users
  FROM dws_user_daily_v
  GROUP BY dt
)
SELECT
  c.dt,
  CAST(c.impressions AS BIGINT) AS impressions,
  CAST(c.clicks AS BIGINT) AS clicks,
  CAST(c.conversions AS BIGINT) AS conversions,
  CAST(ROUND(c.gmv, 5) AS DOUBLE) AS gmv,
  CAST(ROUND(c.cost, 5) AS DOUBLE) AS cost,
  CAST(ROUND(CASE WHEN COALESCE(c.impressions, 0) = 0 THEN 0.0 ELSE c.clicks / c.impressions END, 5) AS DOUBLE) AS ctr,
  CAST(ROUND(CASE WHEN COALESCE(c.clicks, 0) = 0 THEN 0.0 ELSE c.conversions / c.clicks END, 5) AS DOUBLE) AS cvr,
  CAST(ROUND(CASE WHEN COALESCE(c.impressions, 0) = 0 THEN 0.0 ELSE c.gmv * 1000.0 / c.impressions END, 5) AS DOUBLE) AS rpm,
  CAST(ROUND(CASE WHEN COALESCE(u.active_users, 0) = 0 THEN 0.0 ELSE c.gmv / u.active_users END, 5) AS DOUBLE) AS arpu,
  CAST(COALESCE(u.active_users, 0) AS BIGINT) AS active_users,
  CAST(ROUND(CASE WHEN COALESCE(c.cost, 0) = 0 THEN 0.0 ELSE c.gmv / c.cost END, 5) AS DOUBLE) AS roi
FROM campaign_agg c
LEFT JOIN active_users_agg u
  ON c.dt = u.dt
""".strip()
    overview = spark.sql(sql_kpi_overview_daily)
    out_overview = paths.warehouse_table("ads/ads_kpi_overview_daily")
    _write_partitioned_table_or_path(
        spark,
        overview,
        "ads.kpi_overview_daily",
        out_overview,
        dt,
        ["impressions", "clicks", "conversions", "gmv", "cost", "ctr", "cvr", "rpm", "arpu", "active_users", "roi"],
        repartition_n=1,
    )

    if not args.no_dq:
        overview_cnt = overview.count()
        print(f"[DQ] ads_kpi_overview_daily row count for dt={dt}: {overview_cnt}")
        _dq_rate_in_range(overview, "ctr", 0.0, 1.0, "ads_kpi_overview_daily")
        _dq_rate_in_range(overview, "cvr", 0.0, 1.0, "ads_kpi_overview_daily")
        _dq_rate_ge(overview, "roi", 0.0, "ads_kpi_overview_daily")

    # 2) ads.campaign_ranking_daily
    sql_campaign_ranking_daily = """
WITH campaign_base AS (
  SELECT
    dt,
    campaign_id,
    CAST(impressions AS BIGINT) AS impressions,
    CAST(clicks AS BIGINT) AS clicks,
    CAST(conversions AS BIGINT) AS conversions,
    CAST(ROUND(gmv_amount, 5) AS DOUBLE) AS gmv,
    CAST(ROUND(cost, 5) AS DOUBLE) AS cost,
    CAST(ROUND(ctr, 5) AS DOUBLE) AS ctr,
    CAST(ROUND(cvr, 5) AS DOUBLE) AS cvr,
    CAST(ROUND(rpm, 5) AS DOUBLE) AS rpm,
    CAST(ROUND(roi, 5) AS DOUBLE) AS roi
  FROM dws_campaign_daily_v
)
SELECT
  dt,
  campaign_id,
  impressions,
  clicks,
  conversions,
  gmv,
  cost,
  ctr,
  cvr,
  rpm,
  roi,
  CAST(ROW_NUMBER() OVER (PARTITION BY dt ORDER BY gmv DESC, campaign_id ASC) AS INT) AS rank_by_gmv,
  CAST(ROW_NUMBER() OVER (PARTITION BY dt ORDER BY roi DESC, campaign_id ASC) AS INT) AS rank_by_roi
FROM campaign_base
""".strip()
    campaign_ranking = spark.sql(sql_campaign_ranking_daily)
    out_campaign_ranking = paths.warehouse_table("ads/ads_campaign_ranking_daily")
    _write_partitioned_table_or_path(
        spark,
        campaign_ranking,
        "ads.campaign_ranking_daily",
        out_campaign_ranking,
        dt,
        ["campaign_id", "impressions", "clicks", "conversions", "gmv", "cost", "ctr", "cvr", "rpm", "roi", "rank_by_gmv", "rank_by_roi"],
        repartition_n=1,
    )
    if not args.no_dq:
        _dq_pk_rate(campaign_ranking, ["dt", "campaign_id"], "ads_campaign_ranking_daily")

    # 3) ads.advertiser_dashboard_daily
    sql_advertiser_dashboard_daily = """
SELECT
  cur.dt,
  cur.advertiser_id,
  CAST(cur.impressions AS BIGINT) AS impressions,
  CAST(cur.clicks AS BIGINT) AS clicks,
  CAST(cur.conversions AS BIGINT) AS conversions,
  CAST(ROUND(cur.gmv_amount, 5) AS DOUBLE) AS gmv,
  CAST(ROUND(cur.cost, 5) AS DOUBLE) AS cost,
  CAST(ROUND(cur.ctr, 5) AS DOUBLE) AS ctr,
  CAST(ROUND(cur.cvr, 5) AS DOUBLE) AS cvr,
  CAST(ROUND(cur.roi, 5) AS DOUBLE) AS roi,
  CASE
    WHEN prev.prev_gmv IS NULL OR prev.prev_gmv = 0 THEN NULL
    ELSE CAST(ROUND((CAST(cur.gmv_amount AS DOUBLE) - prev.prev_gmv) / prev.prev_gmv, 5) AS DOUBLE)
  END AS revenue_growth_rate,
  CAST(CASE WHEN CAST(cur.cost AS DOUBLE) > 0 THEN 1 ELSE 0 END AS INT) AS retention_flag
FROM dws_advertiser_daily_v cur
LEFT JOIN dws_advertiser_daily_prev_v prev
  ON cur.advertiser_id = prev.advertiser_id
""".strip()
    dash = spark.sql(sql_advertiser_dashboard_daily)
    out_dash = paths.warehouse_table("ads/ads_advertiser_dashboard_daily")
    _write_partitioned_table_or_path(
        spark,
        dash,
        "ads.advertiser_dashboard_daily",
        out_dash,
        dt,
        ["advertiser_id", "impressions", "clicks", "conversions", "gmv", "cost", "ctr", "cvr", "roi", "revenue_growth_rate", "retention_flag"],
        repartition_n=1,
    )

    if not args.no_dq:
        _dq_pk_rate(dash, ["dt", "advertiser_id"], "ads_advertiser_dashboard_daily")
        inf_rows = dash.where(F.col("revenue_growth_rate").isin(float("inf"), float("-inf"))).count()
        print(f"[DQ] ads_advertiser_dashboard_daily growth_rate inf rows={inf_rows}")

    # 4) ads.tag_effectiveness_daily
    sql_tag_effectiveness_daily = """
SELECT
  dt,
  tag_name,
  CAST(ROUND(coverage_rate, 5) AS DOUBLE) AS coverage_rate,
  CAST(tagged_user_cnt AS BIGINT) AS tagged_user_cnt,
  CAST(ROUND(avg_ctr_tagged, 5) AS DOUBLE) AS avg_ctr_tagged,
  CAST(ROUND(avg_cvr_tagged, 5) AS DOUBLE) AS avg_cvr_tagged,
  CAST(ROUND(avg_gmv_tagged, 5) AS DOUBLE) AS avg_gmv_tagged
FROM dws_tag_quality_daily_v
""".strip()
    tag_eff = spark.sql(sql_tag_effectiveness_daily)
    out_tag_eff = paths.warehouse_table("ads/ads_tag_effectiveness_daily")
    _write_partitioned_table_or_path(
        spark,
        tag_eff,
        "ads.tag_effectiveness_daily",
        out_tag_eff,
        dt,
        ["tag_name", "coverage_rate", "tagged_user_cnt", "avg_ctr_tagged", "avg_cvr_tagged", "avg_gmv_tagged"],
        repartition_n=1,
    )

    if not args.no_dq:
        _dq_pk_rate(tag_eff, ["dt", "tag_name"], "ads_tag_effectiveness_daily")
        _dq_rate_in_range(tag_eff, "coverage_rate", 0.0, 1.0, "ads_tag_effectiveness_daily")

    spark.stop()


if __name__ == "__main__":
    main()

