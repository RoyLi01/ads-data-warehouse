from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
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

    Keep parquet fallback so the project remains runnable before every layer is
    fully wired into Hive.
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
    Prefer Hive-style table writes when the target table already exists.

    SQL remains the dominant modeling language, while Python only coordinates the
    write path and compatibility fallback.
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


def _write_non_partitioned_table_or_path(
    spark,
    df: DataFrame,
    table_name: str,
    out_path: Path,
    data_cols: list[str],
) -> None:
    if _table_exists(spark, table_name):
        temp_view = f"_stage_{table_name.replace('.', '_')}"
        select_cols = ", ".join(f"`{c}`" for c in data_cols)
        df.createOrReplaceTempView(temp_view)
        spark.sql(
            f"""
INSERT OVERWRITE TABLE {table_name}
SELECT {select_cols}
FROM {temp_view}
""".strip()
        )
        spark.catalog.dropTempView(temp_view)
        return

    df.write.mode("overwrite").parquet(str(out_path))


def _dq_pk_rate(df: DataFrame, key_cols: list[str], name: str) -> None:
    total = df.count()
    distinct = df.select(*key_cols).distinct().count() if total else 0
    rate = (distinct / total) if total else 1.0
    print(f"[DQ] {name} PK unique rate: {distinct}/{total} = {rate:.2%}")


def _dq_logic_counts(df: DataFrame, name: str) -> None:
    bad_clicks = df.where(F.col("clicks") > F.col("impressions")).count() if "clicks" in df.columns else 0
    bad_convs = df.where(F.col("conversions") > F.col("clicks")).count() if {"conversions", "clicks"}.issubset(df.columns) else 0
    bad_cost = df.where(F.col("cost") < 0).count() if "cost" in df.columns else 0
    bad_gmv = df.where(F.col("gmv_amount") < 0).count() if "gmv_amount" in df.columns else 0
    print(
        f"[DQ] {name} logic: clicks>impressions rows={bad_clicks}, convs>clicks rows={bad_convs}, cost<0 rows={bad_cost}, gmv<0 rows={bad_gmv}"
    )


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
    1) dwd.campaign_day_cost Hive table
    2) warehouse/dwd/dwd_campaign_day_cost parquet

    DWS cost now depends strictly on the DWD layer. If this input is missing,
    users should run DWD first so the cost chain remains ODS -> DWD -> DWS.
    """
    if _table_exists(spark, "dwd.campaign_day_cost"):
        dwd_cost = spark.table("dwd.campaign_day_cost").where(F.col("dt") == dt).select(
            "dt", "campaign_id", F.col("cost").cast("double").alias("cost")
        )
        if dwd_cost.take(1):
            return dwd_cost

    dwd_cost_path = paths.warehouse_table("dwd/dwd_campaign_day_cost")
    if dwd_cost_path.exists():
        try:
            dwd_cost = _read_parquet(spark, dwd_cost_path).where(F.col("dt") == dt).select(
                "dt", "campaign_id", F.col("cost").cast("double").alias("cost")
            )
            if dwd_cost.take(1):
                return dwd_cost
        except Exception:
            # Hive init may create an empty location before this table has real data.
            pass

    raise FileNotFoundError(
        "Missing campaign cost input for DWS. Expected one of: "
        "dwd.campaign_day_cost or warehouse/dwd/dwd_campaign_day_cost. "
        "Please run DWD first."
    )


def _read_dwd_inputs_for_dt(spark, paths: Paths, dt: str) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Read all DWD inputs required by the DWS daily build for a single dt.

    Keeping this resolution in one helper makes `main()` read more like a
    standard orchestrator: resolve inputs once, register views once, then run
    the SQL modeling blocks.
    """
    imp = _read_table_or_parquet(
        spark,
        "dwd.ad_impression_detail",
        paths.warehouse_table("dwd/dwd_ad_impression_detail"),
    ).where(F.col("dt") == dt)
    clk = _read_table_or_parquet(
        spark,
        "dwd.ad_click_detail",
        paths.warehouse_table("dwd/dwd_ad_click_detail"),
    ).where(F.col("dt") == dt)
    cvr = _read_table_or_parquet(
        spark,
        "dwd.ad_conversion_detail",
        paths.warehouse_table("dwd/dwd_ad_conversion_detail"),
    ).where(F.col("dt") == dt)
    campaign_cost = _read_campaign_cost_for_dt(spark, paths, dt)
    return imp, clk, cvr, campaign_cost


def _campaign_advertiser_map(impression: DataFrame, click: DataFrame, dt: str) -> DataFrame:
    """
    Mapping (dt,campaign_id)->advertiser_id from facts, for cost allocation.

    Keep this tiny piece of DataFrame logic in Python because the "prefer
    impression, fallback to click" control flow is clearer here than in SQL.
    """
    imp_map = (
        impression.where(F.col("dt") == dt)
        .select("dt", "campaign_id", "advertiser_id")
        .where(F.col("campaign_id").isNotNull() & F.col("advertiser_id").isNotNull())
        .dropDuplicates(["dt", "campaign_id", "advertiser_id"])
    )
    if imp_map.take(1):
        return imp_map
    return (
        click.where(F.col("dt") == dt)
        .select("dt", "campaign_id", "advertiser_id")
        .where(F.col("campaign_id").isNotNull() & F.col("advertiser_id").isNotNull())
        .dropDuplicates(["dt", "campaign_id", "advertiser_id"])
    )

# Built-in user tag dictionary used by downstream user snapshot and tag quality outputs.
def _tag_dict_rows() -> list[dict]:
    return [
        {"tag_id": "tag_0001", "tag_name": "high_active_7d", "tag_type": "behavior", "rule_desc": "近7天活跃天数>=3（当天有曝光或点击计为活跃）", "update_freq": "daily"},
        {"tag_id": "tag_0002", "tag_name": "high_ctr_7d", "tag_type": "behavior", "rule_desc": "近7天 CTR>=0.04 且 impressions>=50", "update_freq": "daily"},
        {"tag_id": "tag_0003", "tag_name": "high_cvr_7d", "tag_type": "behavior", "rule_desc": "近7天 CVR>=0.06 且 clicks>=10", "update_freq": "daily"},
        {"tag_id": "tag_0004", "tag_name": "high_value_7d", "tag_type": "value", "rule_desc": "近7天 GMV>=300", "update_freq": "daily"},
        {"tag_id": "tag_0005", "tag_name": "ad_clicker", "tag_type": "behavior", "rule_desc": "当天 clicks>=3", "update_freq": "daily"},
        {"tag_id": "tag_0006", "tag_name": "ad_viewer", "tag_type": "behavior", "rule_desc": "当天 impressions>=20", "update_freq": "daily"},
        {"tag_id": "tag_0007", "tag_name": "converter", "tag_type": "behavior", "rule_desc": "当天 conversions>=1", "update_freq": "daily"},
        {"tag_id": "tag_0008", "tag_name": "bounce_user", "tag_type": "risk", "rule_desc": "当天 impressions>0 且 clicks=0 且 conversions=0（只看不点不转）", "update_freq": "daily"},
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
    enable_dq: bool,
) -> None:
    # -------------------- SQL definitions --------------------
    dws_user_daily_today.createOrReplaceTempView("dws_user_daily_today_v")
    dws_user_daily_hist = _read_table_or_parquet(spark, "dws.user_daily", paths.warehouse_table("dws/dws_user_daily"))
    dws_user_daily_hist.createOrReplaceTempView("dws_user_daily_hist_v")

    window_dts = ",".join(_sql_quote(x) for x in _dt_range_7d(dt))
    sql_tag_dict = _build_tag_dict_sql()

    sql_user_tag_snapshot = f"""
WITH day AS (
  SELECT
    dt,
    user_id,
    CAST(imp_cnt AS BIGINT) AS feature_day_impressions,
    CAST(clk_cnt AS BIGINT) AS feature_day_clicks,
    CAST(conv_cnt AS BIGINT) AS feature_day_conversions,
    CAST(gmv_amount AS DOUBLE) AS feature_day_gmv,
    last_event_time AS day_last_event_time
  FROM dws_user_daily_today_v
),
hist AS (
  SELECT
    dt,
    user_id,
    CAST(imp_cnt AS BIGINT) AS imp_cnt,
    CAST(clk_cnt AS BIGINT) AS clk_cnt,
    CAST(conv_cnt AS BIGINT) AS conv_cnt,
    CAST(gmv_amount AS DOUBLE) AS gmv_amount,
    last_event_time
  FROM dws_user_daily_hist_v
  WHERE dt IN ({window_dts})
),
hist_agg AS (
  SELECT
    user_id,
    SUM(imp_cnt) AS feature_7d_impressions,
    SUM(clk_cnt) AS feature_7d_clicks,
    SUM(conv_cnt) AS feature_7d_conversions,
    SUM(COALESCE(gmv_amount, 0.0)) AS feature_7d_gmv,
    SUM(CASE WHEN imp_cnt > 0 OR clk_cnt > 0 THEN 1 ELSE 0 END) AS feature_7d_active_days,
    MAX(last_event_time) AS last_event_time,
    CAST(ROUND(CASE WHEN SUM(imp_cnt) = 0 THEN 0.0 ELSE SUM(clk_cnt) * 1.0 / SUM(imp_cnt) END, 5) AS DOUBLE) AS feature_7d_ctr,
    CAST(ROUND(CASE WHEN SUM(clk_cnt) = 0 THEN 0.0 ELSE SUM(conv_cnt) * 1.0 / SUM(clk_cnt) END, 5) AS DOUBLE) AS feature_7d_cvr
  FROM hist
  GROUP BY user_id
),
base AS (
  SELECT
    '{_sql_string(dt)}' AS dt,
    d.user_id,
    COALESCE(h.feature_7d_impressions, 0) AS feature_7d_impressions,
    COALESCE(h.feature_7d_clicks, 0) AS feature_7d_clicks,
    COALESCE(h.feature_7d_conversions, 0) AS feature_7d_conversions,
    COALESCE(h.feature_7d_gmv, 0.0) AS feature_7d_gmv,
    COALESCE(h.feature_7d_ctr, 0.0) AS feature_7d_ctr,
    COALESCE(h.feature_7d_cvr, 0.0) AS feature_7d_cvr,
    COALESCE(h.feature_7d_active_days, 0) AS feature_7d_active_days,
    COALESCE(d.feature_day_impressions, 0) AS feature_day_impressions,
    COALESCE(d.feature_day_clicks, 0) AS feature_day_clicks,
    COALESCE(d.feature_day_conversions, 0) AS feature_day_conversions,
    COALESCE(d.feature_day_gmv, 0.0) AS feature_day_gmv,
    COALESCE(h.last_event_time, d.day_last_event_time) AS last_event_time
  FROM day d
  LEFT JOIN hist_agg h
    ON d.user_id = h.user_id
)
SELECT
  dt,
  user_id,
  filter(
    array(
      CASE WHEN feature_7d_active_days >= 3 THEN 'high_active_7d' END,
      CASE WHEN feature_7d_ctr >= 0.04 AND feature_7d_impressions >= 50 THEN 'high_ctr_7d' END,
      CASE WHEN feature_7d_cvr >= 0.06 AND feature_7d_clicks >= 10 THEN 'high_cvr_7d' END,
      CASE WHEN feature_7d_gmv >= 300.0 THEN 'high_value_7d' END,
      CASE WHEN feature_day_clicks >= 3 THEN 'ad_clicker' END,
      CASE WHEN feature_day_impressions >= 20 THEN 'ad_viewer' END,
      CASE WHEN feature_day_conversions >= 1 THEN 'converter' END,
      CASE WHEN feature_day_impressions > 0 AND feature_day_clicks = 0 AND feature_day_conversions = 0 THEN 'bounce_user' END
    ),
    x -> x IS NOT NULL
  ) AS tags,
  size(
    filter(
      array(
        CASE WHEN feature_7d_active_days >= 3 THEN 'high_active_7d' END,
        CASE WHEN feature_7d_ctr >= 0.04 AND feature_7d_impressions >= 50 THEN 'high_ctr_7d' END,
        CASE WHEN feature_7d_cvr >= 0.06 AND feature_7d_clicks >= 10 THEN 'high_cvr_7d' END,
        CASE WHEN feature_7d_gmv >= 300.0 THEN 'high_value_7d' END,
        CASE WHEN feature_day_clicks >= 3 THEN 'ad_clicker' END,
        CASE WHEN feature_day_impressions >= 20 THEN 'ad_viewer' END,
        CASE WHEN feature_day_conversions >= 1 THEN 'converter' END,
        CASE WHEN feature_day_impressions > 0 AND feature_day_clicks = 0 AND feature_day_conversions = 0 THEN 'bounce_user' END
      ),
      x -> x IS NOT NULL
    )
  ) AS tag_cnt,
  feature_7d_impressions,
  feature_7d_clicks,
  feature_7d_conversions,
  feature_7d_gmv,
  feature_7d_ctr,
  feature_7d_cvr,
  feature_day_impressions,
  feature_day_clicks,
  feature_day_conversions,
  feature_day_gmv,
  last_event_time
FROM base
""".strip()

    # -------------------- SQL execution --------------------
    tag_dict = spark.sql(sql_tag_dict)
    tag_dict.createOrReplaceTempView("tag_dict_v")
    snapshot = spark.sql(sql_user_tag_snapshot)
    snapshot.createOrReplaceTempView("user_tag_snapshot_v")
    total_user_cnt = snapshot.select("user_id").distinct().count()

    sql_tag_quality_daily = f"""
WITH exploded AS (
  SELECT
    dt,
    user_id,
    feature_7d_ctr,
    feature_7d_cvr,
    feature_7d_gmv,
    EXPLODE_OUTER(tags) AS tag_name
  FROM user_tag_snapshot_v
),
agg AS (
  SELECT
    dt,
    tag_name,
    COUNT(DISTINCT user_id) AS tagged_user_cnt,
    AVG(feature_7d_ctr) AS avg_ctr_tagged,
    AVG(feature_7d_cvr) AS avg_cvr_tagged,
    AVG(feature_7d_gmv) AS avg_gmv_tagged
  FROM exploded
  WHERE tag_name IS NOT NULL
  GROUP BY dt, tag_name
),
all_tags AS (
  SELECT '{_sql_string(dt)}' AS dt, tag_name
  FROM tag_dict_v
)
SELECT
  t.dt,
  t.tag_name,
  CAST(COALESCE(a.tagged_user_cnt, 0) AS BIGINT) AS tagged_user_cnt,
  CAST({_sql_string(str(total_user_cnt))} AS BIGINT) AS total_user_cnt,
  CAST(ROUND(
    CASE
      WHEN CAST({_sql_string(str(total_user_cnt))} AS BIGINT) = 0 THEN 0.0
      ELSE COALESCE(a.tagged_user_cnt, 0) * 1.0 / CAST({_sql_string(str(total_user_cnt))} AS BIGINT)
    END,
    5
  ) AS DOUBLE) AS coverage_rate,
  CAST(ROUND(COALESCE(a.avg_ctr_tagged, 0.0), 5) AS DOUBLE) AS avg_ctr_tagged,
  CAST(ROUND(COALESCE(a.avg_cvr_tagged, 0.0), 5) AS DOUBLE) AS avg_cvr_tagged,
  CAST(COALESCE(a.avg_gmv_tagged, 0.0) AS DOUBLE) AS avg_gmv_tagged
FROM all_tags t
LEFT JOIN agg a
  ON t.dt = a.dt AND t.tag_name = a.tag_name
""".strip()
    quality = spark.sql(sql_tag_quality_daily)

    # -------------------- DQ + writes --------------------
    out_tag_dict = paths.warehouse_table("dws/dws_tag_dict")
    _write_non_partitioned_table_or_path(
        spark,
        tag_dict,
        "dws.tag_dict",
        out_tag_dict,
        ["tag_id", "tag_name", "tag_type", "rule_desc", "update_freq", "create_time"],
    )

    out_snapshot = paths.warehouse_table("dws/dws_user_tag_snapshot")
    _write_partitioned_table_or_path(
        spark,
        snapshot,
        "dws.user_tag_snapshot",
        out_snapshot,
        dt,
        [
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
        ],
        repartition_n=8,
    )

    if enable_dq:
        _dq_pk_rate(snapshot, ["dt", "user_id"], "dws_user_tag_snapshot")
        _dq_rate_ge(snapshot, "tag_cnt", 0, "dws_user_tag_snapshot")
        _dq_rate_in_range(snapshot, "feature_7d_ctr", 0.0, 1.0, "dws_user_tag_snapshot")
        _dq_rate_in_range(snapshot, "feature_7d_cvr", 0.0, 1.0, "dws_user_tag_snapshot")

    out_quality = paths.warehouse_table("dws/dws_tag_quality_daily")
    _write_partitioned_table_or_path(
        spark,
        quality,
        "dws.tag_quality_daily",
        out_quality,
        dt,
        ["tag_name", "tagged_user_cnt", "total_user_cnt", "coverage_rate", "avg_ctr_tagged", "avg_cvr_tagged", "avg_gmv_tagged"],
        repartition_n=1,
    )

    if enable_dq:
        _dq_pk_rate(quality, ["dt", "tag_name"], "dws_tag_quality_daily")
        _dq_rate_in_range(quality, "coverage_rate", 0.0, 1.0, "dws_tag_quality_daily")
        bad_cnt = quality.where(F.col("tagged_user_cnt") > F.col("total_user_cnt")).count()
        print(f"[DQ] dws_tag_quality_daily tagged_user_cnt<=total_user_cnt violated rows={bad_cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DWS aggregates from DWD (incremental by dt).")
    parser.add_argument("--dt", required=True, help="YYYY-MM-DD (single-day incremental run)")
    parser.add_argument("--no_dq", action="store_true", help="Skip DQ prints that trigger expensive counts.")
    args = parser.parse_args()

    dt = args.dt
    paths = Paths(project_root=_project_root())

    spark = build_spark(
        app_name=f"build_dws_dt={dt}",
        extra_conf={
            "spark.pyspark.driver.python": sys.executable,
            "spark.pyspark.python": sys.executable,
        },
    )

    # Python keeps table/path fallback resolution. Core DWS logic below is SQL-first.
    imp, clk, cvr, campaign_cost = _read_dwd_inputs_for_dt(spark, paths, dt)

    imp.createOrReplaceTempView("dwd_imp_v")
    clk.createOrReplaceTempView("dwd_clk_v")
    cvr.createOrReplaceTempView("dwd_cvr_v")
    campaign_cost.createOrReplaceTempView("campaign_cost_v")

    camp_adv_map = _campaign_advertiser_map(imp, clk, dt)
    camp_adv_map.createOrReplaceTempView("camp_adv_map_v")

    # -------------------- SQL definitions --------------------
    sql_campaign_daily = """
WITH keys AS (
  SELECT dt, campaign_id FROM dwd_imp_v
  UNION
  SELECT dt, campaign_id FROM dwd_clk_v
  UNION
  SELECT dt, campaign_id FROM dwd_cvr_v
  UNION
  SELECT dt, campaign_id FROM campaign_cost_v
),
imp_campaign AS (
  SELECT dt, campaign_id, COUNT(1) AS impressions
  FROM dwd_imp_v
  GROUP BY dt, campaign_id
),
clk_campaign AS (
  SELECT dt, campaign_id, COUNT(1) AS clicks
  FROM dwd_clk_v
  GROUP BY dt, campaign_id
),
cvr_campaign AS (
  SELECT
    dt,
    campaign_id,
    COUNT(1) AS conversions,
    SUM(COALESCE(CAST(gmv_amount AS DOUBLE), 0.0)) AS gmv_amount
  FROM dwd_cvr_v
  GROUP BY dt, campaign_id
)
SELECT
  k.dt,
  k.campaign_id,
  CAST(COALESCE(i.impressions, 0) AS BIGINT) AS impressions,
  CAST(COALESCE(c.clicks, 0) AS BIGINT) AS clicks,
  CAST(COALESCE(v.conversions, 0) AS BIGINT) AS conversions,
  CAST(COALESCE(v.gmv_amount, 0.0) AS DOUBLE) AS gmv_amount,
  CAST(COALESCE(cost.cost, 0.0) AS DOUBLE) AS cost,
  CAST(ROUND(CASE WHEN COALESCE(i.impressions, 0) = 0 THEN 0.0 ELSE COALESCE(c.clicks, 0) * 1.0 / i.impressions END, 5) AS DOUBLE) AS ctr,
  CAST(ROUND(CASE WHEN COALESCE(c.clicks, 0) = 0 THEN 0.0 ELSE COALESCE(v.conversions, 0) * 1.0 / c.clicks END, 5) AS DOUBLE) AS cvr,
  CAST(ROUND(CASE WHEN COALESCE(i.impressions, 0) = 0 THEN 0.0 ELSE COALESCE(v.gmv_amount, 0.0) * 1000.0 / i.impressions END, 5) AS DOUBLE) AS rpm,
  CAST(ROUND(CASE WHEN COALESCE(cost.cost, 0.0) = 0 THEN 0.0 ELSE COALESCE(v.gmv_amount, 0.0) / cost.cost END, 5) AS DOUBLE) AS roi
FROM keys k
LEFT JOIN imp_campaign i
  ON k.dt = i.dt AND k.campaign_id = i.campaign_id
LEFT JOIN clk_campaign c
  ON k.dt = c.dt AND k.campaign_id = c.campaign_id
LEFT JOIN cvr_campaign v
  ON k.dt = v.dt AND k.campaign_id = v.campaign_id
LEFT JOIN campaign_cost_v cost
  ON k.dt = cost.dt AND k.campaign_id = cost.campaign_id
""".strip()

    sql_advertiser_daily = """
WITH keys AS (
  SELECT dt, advertiser_id FROM dwd_imp_v WHERE advertiser_id IS NOT NULL
  UNION
  SELECT dt, advertiser_id FROM dwd_clk_v WHERE advertiser_id IS NOT NULL
  UNION
  SELECT dt, advertiser_id FROM dwd_cvr_v WHERE advertiser_id IS NOT NULL
  UNION
  SELECT dt, advertiser_id FROM camp_adv_map_v WHERE advertiser_id IS NOT NULL
),
imp_adv AS (
  SELECT dt, advertiser_id, COUNT(1) AS impressions
  FROM dwd_imp_v
  WHERE advertiser_id IS NOT NULL
  GROUP BY dt, advertiser_id
),
clk_adv AS (
  SELECT dt, advertiser_id, COUNT(1) AS clicks
  FROM dwd_clk_v
  WHERE advertiser_id IS NOT NULL
  GROUP BY dt, advertiser_id
),
cvr_adv AS (
  SELECT
    dt,
    advertiser_id,
    COUNT(1) AS conversions,
    SUM(COALESCE(CAST(gmv_amount AS DOUBLE), 0.0)) AS gmv_amount
  FROM dwd_cvr_v
  WHERE advertiser_id IS NOT NULL
  GROUP BY dt, advertiser_id
),
adv_cnt AS (
  SELECT dt, campaign_id, COUNT(DISTINCT advertiser_id) AS adv_cnt
  FROM camp_adv_map_v
  GROUP BY dt, campaign_id
),
adv_cost AS (
  SELECT
    cost.dt,
    map.advertiser_id,
    SUM(COALESCE(cost.cost, 0.0) / CASE WHEN COALESCE(cnt.adv_cnt, 0) = 0 THEN 1 ELSE cnt.adv_cnt END) AS cost
  FROM campaign_cost_v cost
  LEFT JOIN adv_cnt cnt
    ON cost.dt = cnt.dt AND cost.campaign_id = cnt.campaign_id
  LEFT JOIN camp_adv_map_v map
    ON cost.dt = map.dt AND cost.campaign_id = map.campaign_id
  GROUP BY cost.dt, map.advertiser_id
)
SELECT
  k.dt,
  k.advertiser_id,
  CAST(COALESCE(i.impressions, 0) AS BIGINT) AS impressions,
  CAST(COALESCE(c.clicks, 0) AS BIGINT) AS clicks,
  CAST(COALESCE(v.conversions, 0) AS BIGINT) AS conversions,
  CAST(COALESCE(v.gmv_amount, 0.0) AS DOUBLE) AS gmv_amount,
  CAST(COALESCE(ac.cost, 0.0) AS DOUBLE) AS cost,
  CAST(ROUND(CASE WHEN COALESCE(i.impressions, 0) = 0 THEN 0.0 ELSE COALESCE(c.clicks, 0) * 1.0 / i.impressions END, 5) AS DOUBLE) AS ctr,
  CAST(ROUND(CASE WHEN COALESCE(c.clicks, 0) = 0 THEN 0.0 ELSE COALESCE(v.conversions, 0) * 1.0 / c.clicks END, 5) AS DOUBLE) AS cvr,
  CAST(ROUND(CASE WHEN COALESCE(i.impressions, 0) = 0 THEN 0.0 ELSE COALESCE(v.gmv_amount, 0.0) * 1000.0 / i.impressions END, 5) AS DOUBLE) AS rpm,
  CAST(ROUND(CASE WHEN COALESCE(ac.cost, 0.0) = 0 THEN 0.0 ELSE COALESCE(v.gmv_amount, 0.0) / ac.cost END, 5) AS DOUBLE) AS roi
FROM keys k
LEFT JOIN imp_adv i
  ON k.dt = i.dt AND k.advertiser_id = i.advertiser_id
LEFT JOIN clk_adv c
  ON k.dt = c.dt AND k.advertiser_id = c.advertiser_id
LEFT JOIN cvr_adv v
  ON k.dt = v.dt AND k.advertiser_id = v.advertiser_id
LEFT JOIN adv_cost ac
  ON k.dt = ac.dt AND k.advertiser_id = ac.advertiser_id
""".strip()

    sql_user_daily = """
WITH keys AS (
  SELECT dt, user_id FROM dwd_imp_v
  UNION
  SELECT dt, user_id FROM dwd_clk_v
  UNION
  SELECT dt, user_id FROM dwd_cvr_v
),
imp_user AS (
  SELECT
    dt,
    user_id,
    COUNT(1) AS imp_cnt,
    COUNT(DISTINCT campaign_id) AS imp_campaign_cnt,
    MAX(event_time) AS last_imp_time
  FROM dwd_imp_v
  GROUP BY dt, user_id
),
clk_user AS (
  SELECT
    dt,
    user_id,
    COUNT(1) AS clk_cnt,
    COUNT(DISTINCT campaign_id) AS clk_campaign_cnt,
    MAX(event_time) AS last_clk_time
  FROM dwd_clk_v
  GROUP BY dt, user_id
),
cvr_user AS (
  SELECT
    dt,
    user_id,
    COUNT(1) AS conv_cnt,
    SUM(COALESCE(CAST(gmv_amount AS DOUBLE), 0.0)) AS gmv_amount,
    MAX(conv_time) AS last_conv_time
  FROM dwd_cvr_v
  GROUP BY dt, user_id
)
SELECT
  k.dt,
  k.user_id,
  CAST(COALESCE(i.imp_cnt, 0) AS BIGINT) AS imp_cnt,
  CAST(COALESCE(c.clk_cnt, 0) AS BIGINT) AS clk_cnt,
  CAST(COALESCE(v.conv_cnt, 0) AS BIGINT) AS conv_cnt,
  CAST(COALESCE(v.gmv_amount, 0.0) AS DOUBLE) AS gmv_amount,
  CAST(GREATEST(COALESCE(i.imp_campaign_cnt, 0), COALESCE(c.clk_campaign_cnt, 0)) AS BIGINT) AS active_campaign_cnt,
  GREATEST(i.last_imp_time, c.last_clk_time, v.last_conv_time) AS last_event_time,
  CAST(ROUND(CASE WHEN COALESCE(i.imp_cnt, 0) = 0 THEN 0.0 ELSE COALESCE(c.clk_cnt, 0) * 1.0 / i.imp_cnt END, 5) AS DOUBLE) AS ctr,
  CAST(ROUND(CASE WHEN COALESCE(c.clk_cnt, 0) = 0 THEN 0.0 ELSE COALESCE(v.conv_cnt, 0) * 1.0 / c.clk_cnt END, 5) AS DOUBLE) AS cvr
FROM keys k
LEFT JOIN imp_user i
  ON k.dt = i.dt AND k.user_id = i.user_id
LEFT JOIN clk_user c
  ON k.dt = c.dt AND k.user_id = c.user_id
LEFT JOIN cvr_user v
  ON k.dt = v.dt AND k.user_id = v.user_id
""".strip()

    # -------------------- SQL execution --------------------
    dws_campaign_daily = spark.sql(sql_campaign_daily)
    dws_advertiser_daily = spark.sql(sql_advertiser_daily)
    dws_user_daily = spark.sql(sql_user_daily)

    # -------------------- DQ + writes --------------------
    out_campaign = paths.warehouse_table("dws/dws_campaign_daily")
    _write_partitioned_table_or_path(
        spark,
        dws_campaign_daily,
        "dws.campaign_daily",
        out_campaign,
        dt,
        ["campaign_id", "impressions", "clicks", "conversions", "gmv_amount", "cost", "ctr", "cvr", "rpm", "roi"],
        repartition_n=1,
    )

    if not args.no_dq:
        _dq_pk_rate(dws_campaign_daily, ["dt", "campaign_id"], "dws_campaign_daily")
        _dq_logic_counts(dws_campaign_daily, "dws_campaign_daily")
        _dq_metric_sanity(dws_campaign_daily, "dws_campaign_daily")

    out_adv = paths.warehouse_table("dws/dws_advertiser_daily")
    _write_partitioned_table_or_path(
        spark,
        dws_advertiser_daily,
        "dws.advertiser_daily",
        out_adv,
        dt,
        ["advertiser_id", "impressions", "clicks", "conversions", "gmv_amount", "cost", "ctr", "cvr", "rpm", "roi"],
        repartition_n=1,
    )

    if not args.no_dq:
        _dq_pk_rate(dws_advertiser_daily, ["dt", "advertiser_id"], "dws_advertiser_daily")
        _dq_logic_counts(dws_advertiser_daily, "dws_advertiser_daily")
        _dq_metric_sanity(dws_advertiser_daily, "dws_advertiser_daily")

    out_user = paths.warehouse_table("dws/dws_user_daily")
    _write_partitioned_table_or_path(
        spark,
        dws_user_daily,
        "dws.user_daily",
        out_user,
        dt,
        ["user_id", "imp_cnt", "clk_cnt", "conv_cnt", "gmv_amount", "active_campaign_cnt", "last_event_time", "ctr", "cvr"],
        repartition_n=8,
    )

    if not args.no_dq:
        _dq_pk_rate(dws_user_daily, ["dt", "user_id"], "dws_user_daily")
        bad_ctr = dws_user_daily.where((F.col("ctr") < 0) | (F.col("ctr") > 1)).count()
        bad_cvr = dws_user_daily.where((F.col("cvr") < 0) | (F.col("cvr") > 1)).count()
        bad_gmv = dws_user_daily.where(F.col("gmv_amount") < 0).count()
        print(f"[DQ] dws_user_daily sanity: ctr out-of-range rows={bad_ctr}, cvr out-of-range rows={bad_cvr}, gmv<0 rows={bad_gmv}")

    _build_tags(
        spark=spark,
        paths=paths,
        dt=dt,
        dws_user_daily_today=dws_user_daily,
        enable_dq=(not args.no_dq),
    )

    spark.stop()


if __name__ == "__main__":
    main()

