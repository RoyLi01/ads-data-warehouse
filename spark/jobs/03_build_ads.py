from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark.common.spark_session import build_spark


def _project_root() -> Path:
    # spark/jobs/*.py -> spark/jobs -> spark -> project root
    return Path(__file__).resolve().parents[2]


def _table_path(warehouse_dir: Path, layer: str, table: str) -> Path:
    return warehouse_dir / layer / table


def _read_table(spark, warehouse_dir: Path, layer: str, table: str) -> DataFrame:
    return spark.read.parquet(str(_table_path(warehouse_dir, layer, table)))


def _write_partitioned(df: DataFrame, warehouse_dir: Path, layer: str, table: str) -> None:
    (
        df.write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(_table_path(warehouse_dir, layer, table)))
    )


def _safe_div(num: F.Column, den: F.Column) -> F.Column:
    return F.when(den == 0, F.lit(None)).otherwise(num / den)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ADS layer reports from DWS + DWD.")
    parser.add_argument("--warehouse_dir", default=str(_project_root() / "warehouse"))
    parser.add_argument(
        "--dt",
        default=None,
        help="Optional partition date filter (YYYY-MM-DD). If omitted, process all dt partitions found.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Top N creatives for ads_top_creatives_daily.",
    )
    parser.add_argument(
        "--rank_by",
        choices=["clicks", "roi"],
        default="clicks",
        help="Ranking metric for top creatives (clicks or roi).",
    )
    args = parser.parse_args()

    warehouse_dir = Path(args.warehouse_dir)
    dt_filter = args.dt
    top_n = int(args.top_n)
    rank_by = args.rank_by

    spark = build_spark(app_name="03_build_ads")

    # ===================== Inputs =====================
    # DWS: campaign daily aggregates (already contains cost)
    dws_campaign = _read_table(spark, warehouse_dir, "dws", "dws_ad_campaign_stats_1d")

    # DWD: details used for creative ranking
    dwd_imp = _read_table(spark, warehouse_dir, "dwd", "dwd_ad_impression_detail")
    dwd_clk = _read_table(spark, warehouse_dir, "dwd", "dwd_ad_click_detail")
    dwd_cvr = _read_table(spark, warehouse_dir, "dwd", "dwd_ad_conversion_detail")

    if dt_filter:
        dws_campaign = dws_campaign.where(F.col("dt") == dt_filter)
        dwd_imp = dwd_imp.where(F.col("dt") == dt_filter)
        dwd_clk = dwd_clk.where(F.col("dt") == dt_filter)
        dwd_cvr = dwd_cvr.where(F.col("dt") == dt_filter)

    # ===================== ADS 1) campaign daily report =====================
    # Table: ads_campaign_daily_report
    # Grain: 1 row per (dt, campaign_id)
    #
    # Metrics:
    # - ctr = clicks / imps
    # - cvr = convs / clicks
    # - cpc = cost / clicks
    # - cpm = cost / imps * 1000
    # - roi = gmv / cost
    # - aov = gmv / convs
    ads_campaign_daily_report = (
        dws_campaign.select("dt", "campaign_id", "imps", "clicks", "convs", "gmv", "cost")
        .withColumn("ctr", _safe_div(F.col("clicks").cast("double"), F.col("imps").cast("double")))
        .withColumn("cvr", _safe_div(F.col("convs").cast("double"), F.col("clicks").cast("double")))
        .withColumn("cpc", _safe_div(F.col("cost").cast("double"), F.col("clicks").cast("double")))
        .withColumn("cpm", _safe_div(F.col("cost").cast("double") * F.lit(1000.0), F.col("imps").cast("double")))
        .withColumn("roi", _safe_div(F.col("gmv").cast("double"), F.col("cost").cast("double")))
        .withColumn("aov", _safe_div(F.col("gmv").cast("double"), F.col("convs").cast("double")))
        .select(
            "dt",
            "campaign_id",
            F.col("imps").cast("long").alias("imps"),
            F.col("clicks").cast("long").alias("clicks"),
            F.col("convs").cast("long").alias("convs"),
            F.col("gmv").cast("double").alias("gmv"),
            F.col("cost").cast("double").alias("cost"),
            "ctr",
            "cvr",
            "cpc",
            "cpm",
            "roi",
            "aov",
        )
    )
    _write_partitioned(ads_campaign_daily_report, warehouse_dir, "ads", "ads_campaign_daily_report")

    # ===================== ADS 2) funnel daily =====================
    # Table: ads_funnel_daily
    # Grain: 1 row per dt (whole-site funnel)
    # Metrics:
    # - ctr = clicks / imps
    # - click_to_conv_rate = convs / clicks
    ads_funnel_daily = (
        dws_campaign.groupBy("dt")
        .agg(
            F.sum("imps").alias("imps"),
            F.sum("clicks").alias("clicks"),
            F.sum("convs").alias("convs"),
        )
        .withColumn("ctr", _safe_div(F.col("clicks").cast("double"), F.col("imps").cast("double")))
        .withColumn("click_to_conv_rate", _safe_div(F.col("convs").cast("double"), F.col("clicks").cast("double")))
        .select(
            "dt",
            F.col("imps").cast("long").alias("imps"),
            F.col("clicks").cast("long").alias("clicks"),
            F.col("convs").cast("long").alias("convs"),
            "ctr",
            "click_to_conv_rate",
        )
    )
    _write_partitioned(ads_funnel_daily, warehouse_dir, "ads", "ads_funnel_daily")

    # ===================== ADS 3) top creatives daily =====================
    # Table: ads_top_creatives_daily
    # Grain: 1 row per (dt, rank) for top creatives across the whole site
    #
    # Required fields: dt, rank, creative_id, clicks, ctr, convs, cvr
    #
    # Note on attribution (why creative_id can have convs/gmv):
    # - conversion detail does NOT contain creative_id.
    # - We attribute each conversion to the latest click (last-touch) within the same dt for the same
    #   (user_id, ad_id, campaign_id) where click_time <= conv_time.
    # - This is a simplified offline attribution suitable for interview demos.

    # impressions/clicks by creative
    imp_creative = (
        dwd_imp.groupBy("dt", "campaign_id", "creative_id")
        .agg(F.count(F.lit(1)).alias("imps"))
        .select("dt", "campaign_id", "creative_id", "imps")
    )
    clk_creative = (
        dwd_clk.groupBy("dt", "campaign_id", "creative_id")
        .agg(F.count(F.lit(1)).alias("clicks"))
        .select("dt", "campaign_id", "creative_id", "clicks")
    )

    # last-touch attribution: conv -> latest click (same dt/user/ad/campaign, click_time <= conv_time)
    conv_base = dwd_cvr.select("dt", "conv_id", "conv_time", "user_id", "ad_id", "campaign_id", "gmv_amount")
    clk_base = dwd_clk.select(
        "dt",
        F.col("event_id").alias("click_event_id"),
        F.col("event_time").alias("click_time"),
        "user_id",
        "ad_id",
        "campaign_id",
        "creative_id",
    )

    joined = (
        conv_base.join(clk_base, on=["dt", "user_id", "ad_id", "campaign_id"], how="left")
        .where(F.col("click_time").isNotNull() & (F.col("click_time") <= F.col("conv_time")))
    )
    w_last = Window.partitionBy("dt", "conv_id").orderBy(F.col("click_time").desc())
    conv_attributed = (
        joined.withColumn("rn", F.row_number().over(w_last))
        .where(F.col("rn") == 1)
        .drop("rn")
        .select("dt", "campaign_id", "creative_id", "conv_id", "gmv_amount")
    )

    conv_creative = (
        conv_attributed.groupBy("dt", "campaign_id", "creative_id")
        .agg(
            F.count(F.lit(1)).alias("convs"),
            F.sum(F.coalesce(F.col("gmv_amount"), F.lit(0.0))).alias("gmv"),
        )
        .select("dt", "campaign_id", "creative_id", "convs", "gmv")
    )

    creative_stats = (
        imp_creative.join(clk_creative, on=["dt", "campaign_id", "creative_id"], how="full")
        .join(conv_creative, on=["dt", "campaign_id", "creative_id"], how="full")
        .na.fill({"imps": 0, "clicks": 0, "convs": 0, "gmv": 0.0})
        .withColumn("ctr", _safe_div(F.col("clicks").cast("double"), F.col("imps").cast("double")))
        .withColumn("cvr", _safe_div(F.col("convs").cast("double"), F.col("clicks").cast("double")))
    )

    # For ROI ranking: allocate campaign-level cost to creatives proportionally.
    # - Primary: allocate by click share within campaign/day
    # - Fallback: allocate by impression share if campaign clicks are 0
    campaign_totals = dws_campaign.select(
        "dt",
        "campaign_id",
        F.col("imps").cast("double").alias("campaign_imps"),
        F.col("clicks").cast("double").alias("campaign_clicks"),
        F.col("cost").cast("double").alias("campaign_cost"),
    )

    creative_with_cost = (
        creative_stats.join(campaign_totals, on=["dt", "campaign_id"], how="left")
        .withColumn(
            "alloc_ratio",
            F.when(
                F.col("campaign_clicks") > 0,
                _safe_div(F.col("clicks").cast("double"), F.col("campaign_clicks")),
            ).otherwise(_safe_div(F.col("imps").cast("double"), F.col("campaign_imps"))),
        )
        .withColumn("alloc_cost", F.col("campaign_cost") * F.coalesce(F.col("alloc_ratio"), F.lit(0.0)))
        .withColumn("roi", _safe_div(F.col("gmv").cast("double"), F.col("alloc_cost").cast("double")))
    )

    order_col = F.col("clicks").desc() if rank_by == "clicks" else F.col("roi").desc_nulls_last()
    w_rank = Window.partitionBy("dt").orderBy(order_col, F.col("creative_id").asc())

    ads_top_creatives_daily = (
        creative_with_cost.where(F.col("creative_id").isNotNull())
        .withColumn("rank", F.row_number().over(w_rank))
        .where(F.col("rank") <= top_n)
        .select(
            "dt",
            "rank",
            "creative_id",
            F.col("clicks").cast("long").alias("clicks"),
            "ctr",
            F.col("convs").cast("long").alias("convs"),
            "cvr",
        )
    )

    _write_partitioned(ads_top_creatives_daily, warehouse_dir, "ads", "ads_top_creatives_daily")

    spark.stop()


if __name__ == "__main__":
    main()

