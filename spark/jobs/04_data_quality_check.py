from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

from spark.common.spark_session import build_spark


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _table_path(warehouse_dir: Path, layer: str, table: str) -> Path:
    return warehouse_dir / layer / table


def _read_table(spark, warehouse_dir: Path, layer: str, table: str) -> DataFrame:
    path = _table_path(warehouse_dir, layer, table)
    if not path.exists():
        raise FileNotFoundError(f"Missing table path: {path}")
    return spark.read.parquet(str(path))


def _assert(condition: bool, msg: str, errors: list[str]) -> None:
    if not condition:
        errors.append(msg)


def _col_is_timestamp(df: DataFrame, col: str) -> bool:
    try:
        return isinstance(df.schema[col].dataType, TimestampType)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic data quality checks for the offline ad DW pipeline.")
    parser.add_argument("--warehouse_dir", default=str(_project_root() / "warehouse"))
    parser.add_argument(
        "--dt",
        default=None,
        help="Optional dt filter for checks (YYYY-MM-DD). If omitted, checks all dt partitions.",
    )
    args = parser.parse_args()

    warehouse_dir = Path(args.warehouse_dir)
    dt_filter = args.dt

    spark = build_spark(app_name="04_data_quality_check")
    errors: list[str] = []

    # ----------- ODS checks -----------
    ods_event = _read_table(spark, warehouse_dir, "ods", "ad_event_log")
    ods_conv = _read_table(spark, warehouse_dir, "ods", "conversion_log")
    ods_cost = _read_table(spark, warehouse_dir, "ods", "ad_cost")

    if dt_filter:
        ods_event = ods_event.where(F.col("dt") == dt_filter)
        ods_conv = ods_conv.where(F.col("dt") == dt_filter)
        ods_cost = ods_cost.where(F.col("dt") == dt_filter)

    _assert(ods_event.count() > 0, "ODS ad_event_log is empty for the given scope.", errors)
    _assert(ods_conv.count() >= 0, "ODS conversion_log count check failed (unexpected).", errors)
    _assert(ods_cost.count() > 0, "ODS ad_cost is empty for the given scope.", errors)

    # types
    _assert(_col_is_timestamp(ods_event, "event_time"), "ODS ad_event_log.event_time is not timestamp.", errors)
    _assert(_col_is_timestamp(ods_conv, "conv_time"), "ODS conversion_log.conv_time is not timestamp.", errors)

    # keys not null
    for col in ["dt", "event_id", "campaign_id", "ad_id", "creative_id", "event_type"]:
        n = ods_event.where(F.col(col).isNull()).count()
        _assert(n == 0, f"ODS ad_event_log has NULL {col}: {n}", errors)

    # cost non-negative
    bad_cost = ods_cost.where(F.col("cost") < 0).count()
    _assert(bad_cost == 0, f"ODS ad_cost has negative cost rows: {bad_cost}", errors)

    # ----------- DWD checks -----------
    dwd_imp = _read_table(spark, warehouse_dir, "dwd", "dwd_ad_impression_detail")
    dwd_clk = _read_table(spark, warehouse_dir, "dwd", "dwd_ad_click_detail")
    dwd_cvr = _read_table(spark, warehouse_dir, "dwd", "dwd_ad_conversion_detail")

    if dt_filter:
        dwd_imp = dwd_imp.where(F.col("dt") == dt_filter)
        dwd_clk = dwd_clk.where(F.col("dt") == dt_filter)
        dwd_cvr = dwd_cvr.where(F.col("dt") == dt_filter)

    _assert(_col_is_timestamp(dwd_imp, "event_time"), "DWD impression_detail.event_time is not timestamp.", errors)
    _assert(_col_is_timestamp(dwd_clk, "event_time"), "DWD click_detail.event_time is not timestamp.", errors)
    _assert(_col_is_timestamp(dwd_cvr, "conv_time"), "DWD conversion_detail.conv_time is not timestamp.", errors)

    # dedup uniqueness (soft check)
    dup_imp = dwd_imp.groupBy("dt", "event_id").count().where(F.col("count") > 1).count()
    dup_clk = dwd_clk.groupBy("dt", "event_id").count().where(F.col("count") > 1).count()
    dup_cvr = dwd_cvr.groupBy("dt", "conv_id").count().where(F.col("count") > 1).count()
    _assert(dup_imp == 0, f"DWD impression_detail has duplicated (dt,event_id) groups: {dup_imp}", errors)
    _assert(dup_clk == 0, f"DWD click_detail has duplicated (dt,event_id) groups: {dup_clk}", errors)
    _assert(dup_cvr == 0, f"DWD conversion_detail has duplicated (dt,conv_id) groups: {dup_cvr}", errors)

    # ----------- DWS checks -----------
    dws_campaign = _read_table(spark, warehouse_dir, "dws", "dws_ad_campaign_stats_1d")
    dws_ad = _read_table(spark, warehouse_dir, "dws", "dws_ad_ad_stats_1d")

    if dt_filter:
        dws_campaign = dws_campaign.where(F.col("dt") == dt_filter)
        dws_ad = dws_ad.where(F.col("dt") == dt_filter)

    _assert(dws_campaign.count() > 0, "DWS dws_ad_campaign_stats_1d is empty for the given scope.", errors)
    _assert(dws_ad.count() > 0, "DWS dws_ad_ad_stats_1d is empty for the given scope.", errors)

    # basic metric sanity
    bad_metrics = (
        dws_campaign.where((F.col("imps") < 0) | (F.col("clicks") < 0) | (F.col("convs") < 0) | (F.col("gmv") < 0) | (F.col("cost") < 0))
        .count()
    )
    _assert(bad_metrics == 0, f"DWS campaign stats has negative metrics rows: {bad_metrics}", errors)

    # ----------- ADS checks -----------
    ads_campaign = _read_table(spark, warehouse_dir, "ads", "ads_campaign_daily_report")
    ads_funnel = _read_table(spark, warehouse_dir, "ads", "ads_funnel_daily")
    ads_top = _read_table(spark, warehouse_dir, "ads", "ads_top_creatives_daily")

    if dt_filter:
        ads_campaign = ads_campaign.where(F.col("dt") == dt_filter)
        ads_funnel = ads_funnel.where(F.col("dt") == dt_filter)
        ads_top = ads_top.where(F.col("dt") == dt_filter)

    _assert(ads_campaign.count() > 0, "ADS campaign_daily_report is empty for the given scope.", errors)
    _assert(ads_funnel.count() > 0, "ADS funnel_daily is empty for the given scope.", errors)
    _assert(ads_top.count() > 0, "ADS top_creatives_daily is empty for the given scope.", errors)

    # KPI range sanity (allow NULLs)
    out_of_range = (
        ads_campaign.where(
            (F.col("ctr").isNotNull() & ((F.col("ctr") < 0) | (F.col("ctr") > 1)))
            | (F.col("cvr").isNotNull() & ((F.col("cvr") < 0) | (F.col("cvr") > 1)))
        ).count()
    )
    _assert(out_of_range == 0, f"ADS campaign_daily_report has out-of-range ctr/cvr rows: {out_of_range}", errors)

    if errors:
        print("DATA QUALITY CHECK FAILED:")
        for e in errors:
            print(f"- {e}")
        spark.stop()
        raise SystemExit(1)

    print("DATA QUALITY CHECK PASSED.")
    spark.stop()


if __name__ == "__main__":
    main()

