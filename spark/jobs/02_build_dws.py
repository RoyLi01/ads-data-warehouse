from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

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


def _salt_col(key_cols: list[str], num_salts: int) -> F.Column:
    """
    Deterministic salt in [0, num_salts).

    We salt (add a small random-like bucket) to split hot keys across many reducers.
    Use hash of stable columns to keep deterministic results across runs.
    """
    if num_salts <= 1:
        return F.lit(0)
    return F.pmod(F.abs(F.hash(*[F.col(c) for c in key_cols])), F.lit(int(num_salts)))


def _salted_two_stage_agg(
    df: DataFrame,
    group_cols: list[str],
    salt_on_cols: list[str],
    num_salts: int,
    agg_exprs: dict[str, F.Column],
) -> DataFrame:
    """
    Two-stage aggregation with salting to mitigate data skew on group keys.

    Principle (why it helps):
    - If one key (e.g. a hot campaign_id) has massive rows, a single reducer/task may become a straggler.
    - Add a salt bucket so that the hot key is split into N sub-keys (campaign_id, salt),
      enabling parallel partial aggregation.
    - Then do a second aggregation to merge partials back to the original key.
    """
    salted = df.withColumn("_salt", _salt_col(salt_on_cols, num_salts))
    stage1 = salted.groupBy(*group_cols, "_salt").agg(*[expr.alias(name) for name, expr in agg_exprs.items()])
    # second stage: sum up partial results
    stage2_exprs = [F.sum(F.col(name)).alias(name) for name in agg_exprs.keys()]
    return stage1.groupBy(*group_cols).agg(*stage2_exprs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DWS daily stats from DWD detail tables.")
    parser.add_argument("--warehouse_dir", default=str(_project_root() / "warehouse"))
    parser.add_argument(
        "--dt",
        default=None,
        help="Optional partition date filter (YYYY-MM-DD). If omitted, process all dt partitions found.",
    )
    parser.add_argument(
        "--num_salts",
        type=int,
        default=16,
        help="Number of salt buckets used in skew-mitigation two-stage aggregation.",
    )
    args = parser.parse_args()

    warehouse_dir = Path(args.warehouse_dir)
    dt_filter = args.dt
    num_salts = int(args.num_salts)

    spark = build_spark(app_name="02_build_dws")

    # ===================== Inputs =====================
    # DWD detail tables
    imp = _read_table(spark, warehouse_dir, "dwd", "dwd_ad_impression_detail")
    clk = _read_table(spark, warehouse_dir, "dwd", "dwd_ad_click_detail")
    cvr = _read_table(spark, warehouse_dir, "dwd", "dwd_ad_conversion_detail")
    # ODS cost (already dt + campaign_id aggregated)
    cost = _read_table(spark, warehouse_dir, "ods", "ad_cost")

    if dt_filter:
        imp = imp.where(F.col("dt") == dt_filter)
        clk = clk.where(F.col("dt") == dt_filter)
        cvr = cvr.where(F.col("dt") == dt_filter)
        cost = cost.where(F.col("dt") == dt_filter)

    # ===================== DWS 1) campaign_id daily stats =====================
    # Table: dws_ad_campaign_stats_1d
    # Grain: 1 row per (dt, campaign_id)
    #
    # Skew mitigation:
    # - campaign_id contains a hot key (cmp_hot_0001) with far more rows.
    # - We use "salting two-stage aggregation" for event-side metrics (imps/clicks),
    #   splitting each (dt,campaign_id) into (dt,campaign_id,salt) buckets for parallel aggregation,
    #   then summing partials back.

    # event-side counts (imps/clicks) from DWD event detail tables
    imp_for_campaign = imp.select("dt", "campaign_id", "event_id")
    clk_for_campaign = clk.select("dt", "campaign_id", "event_id")

    imp_counts = _salted_two_stage_agg(
        df=imp_for_campaign,
        group_cols=["dt", "campaign_id"],
        salt_on_cols=["campaign_id", "event_id"],
        num_salts=num_salts,
        agg_exprs={"imps": F.count(F.lit(1))},
    )

    clk_counts = _salted_two_stage_agg(
        df=clk_for_campaign,
        group_cols=["dt", "campaign_id"],
        salt_on_cols=["campaign_id", "event_id"],
        num_salts=num_salts,
        agg_exprs={"clicks": F.count(F.lit(1))},
    )

    # conversion-side metrics (convs, gmv) from conversion detail
    cvr_campaign = cvr.select("dt", "campaign_id", "conv_id", "gmv_amount")
    cvr_agg_campaign = _salted_two_stage_agg(
        df=cvr_campaign,
        group_cols=["dt", "campaign_id"],
        salt_on_cols=["campaign_id", "conv_id"],
        num_salts=max(1, num_salts // 2),
        agg_exprs={
            "convs": F.count(F.lit(1)),
            "gmv": F.sum(F.coalesce(F.col("gmv_amount"), F.lit(0.0))),
        },
    )

    # join all components + cost
    # Broadcast join note:
    # - ad_cost is aggregated by (dt, campaign_id) and is typically much smaller than event details.
    # - Broadcasting it avoids shuffling the large side during the join (common optimization in ad DW).
    dws_ad_campaign_stats_1d = (
        imp_counts.join(clk_counts, on=["dt", "campaign_id"], how="full")
        .join(cvr_agg_campaign, on=["dt", "campaign_id"], how="full")
        .join(
            F.broadcast(cost.select("dt", "campaign_id", F.col("cost").cast("double").alias("cost"))),
            on=["dt", "campaign_id"],
            how="left",
        )
        .na.fill({"imps": 0, "clicks": 0, "convs": 0, "gmv": 0.0, "cost": 0.0})
        .select(
            "dt",
            "campaign_id",
            F.col("imps").cast("long").alias("imps"),
            F.col("clicks").cast("long").alias("clicks"),
            F.col("convs").cast("long").alias("convs"),
            F.col("gmv").cast("double").alias("gmv"),
            F.col("cost").cast("double").alias("cost"),
        )
    )

    _write_partitioned(dws_ad_campaign_stats_1d, warehouse_dir, "dws", "dws_ad_campaign_stats_1d")

    # ===================== DWS 2) ad_id daily stats =====================
    # Table: dws_ad_ad_stats_1d
    # Grain: 1 row per (dt, ad_id)
    #
    # Notes:
    # - ad_id can also be skewed (less obvious than hot campaign), but we reuse the same salted two-stage pattern.

    imp_for_ad = imp.select("dt", "ad_id", "event_id")
    clk_for_ad = clk.select("dt", "ad_id", "event_id")
    cvr_for_ad = cvr.select("dt", "ad_id", "conv_id", "gmv_amount")

    imp_ad = _salted_two_stage_agg(
        df=imp_for_ad,
        group_cols=["dt", "ad_id"],
        salt_on_cols=["ad_id", "event_id"],
        num_salts=num_salts,
        agg_exprs={"imps": F.count(F.lit(1))},
    )
    clk_ad = _salted_two_stage_agg(
        df=clk_for_ad,
        group_cols=["dt", "ad_id"],
        salt_on_cols=["ad_id", "event_id"],
        num_salts=num_salts,
        agg_exprs={"clicks": F.count(F.lit(1))},
    )
    cvr_ad = _salted_two_stage_agg(
        df=cvr_for_ad,
        group_cols=["dt", "ad_id"],
        salt_on_cols=["ad_id", "conv_id"],
        num_salts=max(1, num_salts // 2),
        agg_exprs={
            "convs": F.count(F.lit(1)),
            "gmv": F.sum(F.coalesce(F.col("gmv_amount"), F.lit(0.0))),
        },
    )

    dws_ad_ad_stats_1d = (
        imp_ad.join(clk_ad, on=["dt", "ad_id"], how="full")
        .join(cvr_ad, on=["dt", "ad_id"], how="full")
        .na.fill({"imps": 0, "clicks": 0, "convs": 0, "gmv": 0.0})
        .select(
            "dt",
            "ad_id",
            F.col("imps").cast("long").alias("imps"),
            F.col("clicks").cast("long").alias("clicks"),
            F.col("convs").cast("long").alias("convs"),
            F.col("gmv").cast("double").alias("gmv"),
        )
    )

    _write_partitioned(dws_ad_ad_stats_1d, warehouse_dir, "dws", "dws_ad_ad_stats_1d")

    spark.stop()


if __name__ == "__main__":
    main()

