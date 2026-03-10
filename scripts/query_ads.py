from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from common.spark_session import build_spark


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query ADS outputs with Spark SQL.")
    parser.add_argument("--dt", required=True, help="YYYY-MM-DD")
    parser.add_argument("--warehouse_dir", default=str(_project_root() / "warehouse"), help="Warehouse base dir")
    parser.add_argument("--top_n", type=int, default=10, help="Top N rows for ranking queries")
    args = parser.parse_args()

    dt = args.dt
    warehouse_dir = Path(args.warehouse_dir)
    top_n = int(args.top_n)

    # Use the same Spark config as jobs (AQE on), and pin python executable.
    spark = build_spark(
        app_name=f"query_ads_dt={dt}",
        extra_conf={
            "spark.pyspark.driver.python": sys.executable,
            "spark.pyspark.python": sys.executable,
        },
    )

    # Register views (ADS v2)
    (spark.read.parquet(str(warehouse_dir / "ads" / "ads_kpi_overview_daily"))).createOrReplaceTempView(
        "ads_kpi_overview_daily"
    )
    (spark.read.parquet(str(warehouse_dir / "ads" / "ads_campaign_ranking_daily"))).createOrReplaceTempView(
        "ads_campaign_ranking_daily"
    )
    (spark.read.parquet(str(warehouse_dir / "ads" / "ads_advertiser_dashboard_daily"))).createOrReplaceTempView(
        "ads_advertiser_dashboard_daily"
    )
    (spark.read.parquet(str(warehouse_dir / "ads" / "ads_tag_effectiveness_daily"))).createOrReplaceTempView(
        "ads_tag_effectiveness_daily"
    )

    print("=== ads_kpi_overview_daily ===")
    spark.sql(
        f"""
        select *
        from ads_kpi_overview_daily
        where dt = '{dt}'
        """
    ).show(truncate=False)

    print(f"=== ads_campaign_ranking_daily (top {top_n} by gmv) ===")
    spark.sql(
        f"""
        select
          dt, campaign_id,
          impressions, clicks, conversions, gmv, cost, ctr, cvr, rpm, roi,
          rank_by_gmv, rank_by_roi
        from ads_campaign_ranking_daily
        where dt = '{dt}'
        order by rank_by_gmv asc
        limit {top_n}
        """
    ).show(truncate=False)

    print(f"=== ads_campaign_ranking_daily (top {top_n} by roi) ===")
    spark.sql(
        f"""
        select
          dt, campaign_id,
          impressions, clicks, conversions, gmv, cost, ctr, cvr, rpm, roi,
          rank_by_gmv, rank_by_roi
        from ads_campaign_ranking_daily
        where dt = '{dt}'
        order by rank_by_roi asc
        limit {top_n}
        """
    ).show(truncate=False)

    print(f"=== ads_advertiser_dashboard_daily (top {top_n} by gmv) ===")
    spark.sql(
        f"""
        select
          dt, advertiser_id,
          impressions, clicks, conversions, gmv, cost, ctr, cvr, roi,
          revenue_growth_rate, retention_flag
        from ads_advertiser_dashboard_daily
        where dt = '{dt}'
        order by gmv desc
        limit {top_n}
        """
    ).show(truncate=False)

    print("=== ads_tag_effectiveness_daily (order by coverage_rate) ===")
    spark.sql(
        f"""
        select
          dt, tag_name, coverage_rate, tagged_user_cnt,
          avg_ctr_tagged, avg_cvr_tagged, avg_gmv_tagged
        from ads_tag_effectiveness_daily
        where dt = '{dt}'
        order by coverage_rate desc, tag_name asc
        """
    ).show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()

