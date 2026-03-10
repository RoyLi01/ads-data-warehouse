from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

from common.spark_session import build_spark


def _project_root() -> Path:
    # jobs/*.py -> jobs -> project root
    return Path(__file__).resolve().parents[1]


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


def _dq_ratio_in_range(df: DataFrame, col: str, lo: float, hi: float, name: str, errors: list[str]) -> None:
    if col not in df.columns:
        return
    bad = df.where(F.col(col).isNotNull() & ((F.col(col) < F.lit(lo)) | (F.col(col) > F.lit(hi)))).count()
    _assert(bad == 0, f"{name} has out-of-range {col} rows: {bad}", errors)


def _dq_non_negative(df: DataFrame, cols: list[str], name: str, errors: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            bad = df.where(F.col(c) < 0).count()
            _assert(bad == 0, f"{name} has negative {c} rows: {bad}", errors)


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

    spark = build_spark(app_name="dq_check")
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

    _assert(_col_is_timestamp(ods_event, "event_time"), "ODS ad_event_log.event_time is not timestamp.", errors)
    _assert(_col_is_timestamp(ods_conv, "conv_time"), "ODS conversion_log.conv_time is not timestamp.", errors)

    for col in ["dt", "event_id", "campaign_id", "ad_id", "creative_id", "event_type"]:
        if col in ods_event.columns:
            n = ods_event.where(F.col(col).isNull()).count()
            _assert(n == 0, f"ODS ad_event_log has NULL {col}: {n}", errors)

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

    dup_imp = dwd_imp.groupBy("dt", "event_id").count().where(F.col("count") > 1).count()
    dup_clk = dwd_clk.groupBy("dt", "event_id").count().where(F.col("count") > 1).count()
    dup_cvr = dwd_cvr.groupBy("dt", "conv_id").count().where(F.col("count") > 1).count()
    _assert(dup_imp == 0, f"DWD impression_detail has duplicated (dt,event_id) groups: {dup_imp}", errors)
    _assert(dup_clk == 0, f"DWD click_detail has duplicated (dt,event_id) groups: {dup_clk}", errors)
    _assert(dup_cvr == 0, f"DWD conversion_detail has duplicated (dt,conv_id) groups: {dup_cvr}", errors)

    # ----------- DWS checks -----------
    dws_campaign = _read_table(spark, warehouse_dir, "dws", "dws_campaign_daily")
    dws_adv = _read_table(spark, warehouse_dir, "dws", "dws_advertiser_daily")
    dws_user = _read_table(spark, warehouse_dir, "dws", "dws_user_daily")
    dws_tag_snap = _read_table(spark, warehouse_dir, "dws", "dws_user_tag_snapshot")
    dws_tag_quality = _read_table(spark, warehouse_dir, "dws", "dws_tag_quality_daily")

    if dt_filter:
        dws_campaign = dws_campaign.where(F.col("dt") == dt_filter)
        dws_adv = dws_adv.where(F.col("dt") == dt_filter)
        dws_user = dws_user.where(F.col("dt") == dt_filter)
        dws_tag_snap = dws_tag_snap.where(F.col("dt") == dt_filter)
        dws_tag_quality = dws_tag_quality.where(F.col("dt") == dt_filter)

    _assert(dws_campaign.count() > 0, "DWS dws_campaign_daily is empty for the given scope.", errors)
    _assert(dws_adv.count() >= 0, "DWS dws_advertiser_daily count check failed (unexpected).", errors)
    _assert(dws_user.count() > 0, "DWS dws_user_daily is empty for the given scope.", errors)

    _dq_non_negative(dws_campaign, ["impressions", "clicks", "conversions", "gmv_amount", "cost"], "DWS dws_campaign_daily", errors)
    _dq_ratio_in_range(dws_campaign, "ctr", 0.0, 1.0, "DWS dws_campaign_daily", errors)
    _dq_ratio_in_range(dws_campaign, "cvr", 0.0, 1.0, "DWS dws_campaign_daily", errors)
    if "roi" in dws_campaign.columns:
        bad_roi = dws_campaign.where(F.col("roi") < 0).count()
        _assert(bad_roi == 0, f"DWS dws_campaign_daily has negative roi rows: {bad_roi}", errors)

    _dq_ratio_in_range(dws_adv, "ctr", 0.0, 1.0, "DWS dws_advertiser_daily", errors)
    _dq_ratio_in_range(dws_adv, "cvr", 0.0, 1.0, "DWS dws_advertiser_daily", errors)

    _dq_ratio_in_range(dws_user, "ctr", 0.0, 1.0, "DWS dws_user_daily", errors)
    _dq_ratio_in_range(dws_user, "cvr", 0.0, 1.0, "DWS dws_user_daily", errors)
    _dq_non_negative(dws_user, ["imp_cnt", "clk_cnt", "conv_cnt", "gmv_amount"], "DWS dws_user_daily", errors)

    _dq_ratio_in_range(dws_tag_quality, "coverage_rate", 0.0, 1.0, "DWS dws_tag_quality_daily", errors)
    if {"tagged_user_cnt", "total_user_cnt"}.issubset(set(dws_tag_quality.columns)):
        bad_cov = dws_tag_quality.where(F.col("tagged_user_cnt") > F.col("total_user_cnt")).count()
        _assert(bad_cov == 0, f"DWS dws_tag_quality_daily has tagged_user_cnt>total_user_cnt rows: {bad_cov}", errors)

    # ----------- ADS checks -----------
    ads_overview = _read_table(spark, warehouse_dir, "ads", "ads_kpi_overview_daily")
    ads_campaign_rank = _read_table(spark, warehouse_dir, "ads", "ads_campaign_ranking_daily")
    ads_adv_dash = _read_table(spark, warehouse_dir, "ads", "ads_advertiser_dashboard_daily")
    ads_tag_eff = _read_table(spark, warehouse_dir, "ads", "ads_tag_effectiveness_daily")

    if dt_filter:
        ads_overview = ads_overview.where(F.col("dt") == dt_filter)
        ads_campaign_rank = ads_campaign_rank.where(F.col("dt") == dt_filter)
        ads_adv_dash = ads_adv_dash.where(F.col("dt") == dt_filter)
        ads_tag_eff = ads_tag_eff.where(F.col("dt") == dt_filter)

    _assert(ads_overview.count() > 0, "ADS ads_kpi_overview_daily is empty for the given scope.", errors)
    _assert(ads_campaign_rank.count() >= 0, "ADS ads_campaign_ranking_daily count check failed (unexpected).", errors)
    _assert(ads_adv_dash.count() >= 0, "ADS ads_advertiser_dashboard_daily count check failed (unexpected).", errors)
    _assert(ads_tag_eff.count() >= 0, "ADS ads_tag_effectiveness_daily count check failed (unexpected).", errors)

    _dq_ratio_in_range(ads_overview, "ctr", 0.0, 1.0, "ADS ads_kpi_overview_daily", errors)
    _dq_ratio_in_range(ads_overview, "cvr", 0.0, 1.0, "ADS ads_kpi_overview_daily", errors)
    _dq_non_negative(ads_overview, ["impressions", "clicks", "conversions", "gmv", "cost", "active_users"], "ADS ads_kpi_overview_daily", errors)

    _dq_ratio_in_range(ads_campaign_rank, "ctr", 0.0, 1.0, "ADS ads_campaign_ranking_daily", errors)
    _dq_ratio_in_range(ads_campaign_rank, "cvr", 0.0, 1.0, "ADS ads_campaign_ranking_daily", errors)

    _dq_ratio_in_range(ads_tag_eff, "coverage_rate", 0.0, 1.0, "ADS ads_tag_effectiveness_daily", errors)

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

