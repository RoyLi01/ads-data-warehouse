from __future__ import annotations

"""
Initialize Hive databases and external tables for the local offline DW project.

Design choice:
- Hive manages metadata/catalog only
- Physical data stays in the existing warehouse/... Parquet directories
- Most day-level tables are partitioned by dt
"""

import argparse
from pathlib import Path

from common.spark_session import build_spark


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sql_string(s: str) -> str:
    return s.replace("'", "''")


def _create_database(spark, db_name: str) -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    print(f"[HIVE] Ensured database {db_name}")


def _create_external_parquet_table(
    spark,
    table_name: str,
    columns_sql: str,
    location: Path,
    partitioned_by_dt: bool,
) -> None:
    location.mkdir(parents=True, exist_ok=True)
    partition_sql = "PARTITIONED BY (dt STRING)" if partitioned_by_dt else ""
    spark.sql(
        f"""
CREATE TABLE IF NOT EXISTS {table_name} (
  {columns_sql}
)
USING PARQUET
{partition_sql}
LOCATION '{_sql_string(str(location))}'
""".strip()
    )
    print(f"[HIVE] Ensured table {table_name} -> {location}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Hive databases and external Parquet tables.")
    parser.add_argument("--warehouse_dir", default=str(_project_root() / "warehouse"), help="Warehouse base dir")
    args = parser.parse_args()

    warehouse_dir = Path(args.warehouse_dir)
    spark = build_spark(app_name="init_hive")

    for db in ["ods", "dwd", "dws", "ads"]:
        _create_database(spark, db)

    # -------------------- ODS --------------------
    _create_external_parquet_table(
        spark,
        "ods.ad_event_log",
        """
event_id STRING,
event_time TIMESTAMP,
user_id STRING,
ad_id STRING,
campaign_id STRING,
creative_id STRING,
site_id STRING,
event_type STRING,
device STRING,
session_id STRING,
ad_slot_id STRING,
page_id STRING,
is_valid INT,
ip STRING,
user_agent STRING
""".strip(),
        warehouse_dir / "ods" / "ad_event_log",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "ods.conversion_log",
        """
conv_id STRING,
conv_time TIMESTAMP,
user_id STRING,
ad_id STRING,
campaign_id STRING,
order_id STRING,
gmv_amount DOUBLE
""".strip(),
        warehouse_dir / "ods" / "conversion_log",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "ods.ad_cost",
        """
campaign_id STRING,
cost DOUBLE
""".strip(),
        warehouse_dir / "ods" / "ad_cost",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "ods.user_profile",
        """
user_id STRING,
gender STRING,
age INT,
region STRING,
device_type STRING,
register_dt DATE
""".strip(),
        warehouse_dir / "ods" / "user_profile",
        partitioned_by_dt=False,
    )
    _create_external_parquet_table(
        spark,
        "ods.ad_meta",
        """
ad_id STRING,
campaign_id STRING,
advertiser_id STRING,
ad_type STRING,
landing_type STRING,
product_id STRING,
start_dt DATE,
end_dt DATE
""".strip(),
        warehouse_dir / "ods" / "ad_meta",
        partitioned_by_dt=False,
    )
    _create_external_parquet_table(
        spark,
        "ods.ad_slot",
        """
ad_slot_id STRING,
slot_type STRING,
app STRING,
position STRING,
price_factor DOUBLE
""".strip(),
        warehouse_dir / "ods" / "ad_slot",
        partitioned_by_dt=False,
    )

    # -------------------- DWD --------------------
    _create_external_parquet_table(
        spark,
        "dwd.dim_user",
        """
user_id STRING,
gender STRING,
age INT,
region STRING,
device_type STRING,
register_dt DATE
""".strip(),
        warehouse_dir / "dwd" / "dim" / "dwd_dim_user",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dwd.dim_ad",
        """
ad_id STRING,
campaign_id STRING,
advertiser_id STRING,
ad_type STRING,
landing_type STRING,
product_id STRING,
start_dt DATE,
end_dt DATE
""".strip(),
        warehouse_dir / "dwd" / "dim" / "dwd_dim_ad",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dwd.dim_ad_slot",
        """
ad_slot_id STRING,
slot_type STRING,
app STRING,
position STRING,
price_factor DOUBLE
""".strip(),
        warehouse_dir / "dwd" / "dim" / "dwd_dim_ad_slot",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dwd.ad_impression_detail",
        """
event_id STRING,
event_time TIMESTAMP,
user_id STRING,
session_id STRING,
ad_id STRING,
campaign_id STRING,
creative_id STRING,
advertiser_id STRING,
ad_type STRING,
landing_type STRING,
product_id STRING,
site_id STRING,
page_id STRING,
ad_slot_id STRING,
slot_type STRING,
app STRING,
position STRING,
price_factor DOUBLE,
device STRING,
user_agent STRING,
ip STRING,
event_type STRING
""".strip(),
        warehouse_dir / "dwd" / "dwd_ad_impression_detail",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dwd.ad_click_detail",
        """
event_id STRING,
event_time TIMESTAMP,
user_id STRING,
session_id STRING,
ad_id STRING,
campaign_id STRING,
creative_id STRING,
advertiser_id STRING,
ad_type STRING,
landing_type STRING,
product_id STRING,
site_id STRING,
page_id STRING,
ad_slot_id STRING,
slot_type STRING,
app STRING,
position STRING,
price_factor DOUBLE,
device STRING,
user_agent STRING,
ip STRING,
event_type STRING
""".strip(),
        warehouse_dir / "dwd" / "dwd_ad_click_detail",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dwd.ad_conversion_detail",
        """
conv_id STRING,
conv_time TIMESTAMP,
user_id STRING,
ad_id STRING,
campaign_id STRING,
advertiser_id STRING,
product_id STRING,
order_id STRING,
gmv_amount DOUBLE
""".strip(),
        warehouse_dir / "dwd" / "dwd_ad_conversion_detail",
        partitioned_by_dt=True,
    )

    # DWD daily campaign cost table used by downstream DWS cost aggregation.
    _create_external_parquet_table(
        spark,
        "dwd.campaign_day_cost",
        """
campaign_id STRING,
cost DOUBLE
""".strip(),
        warehouse_dir / "dwd" / "dwd_campaign_day_cost",
        partitioned_by_dt=True,
    )

    # -------------------- DWS --------------------
    _create_external_parquet_table(
        spark,
        "dws.campaign_daily",
        """
campaign_id STRING,
impressions BIGINT,
clicks BIGINT,
conversions BIGINT,
gmv_amount DOUBLE,
cost DOUBLE,
ctr DOUBLE,
cvr DOUBLE,
rpm DOUBLE,
roi DOUBLE
""".strip(),
        warehouse_dir / "dws" / "dws_campaign_daily",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dws.advertiser_daily",
        """
advertiser_id STRING,
impressions BIGINT,
clicks BIGINT,
conversions BIGINT,
gmv_amount DOUBLE,
cost DOUBLE,
ctr DOUBLE,
cvr DOUBLE,
rpm DOUBLE,
roi DOUBLE
""".strip(),
        warehouse_dir / "dws" / "dws_advertiser_daily",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dws.user_daily",
        """
user_id STRING,
imp_cnt BIGINT,
clk_cnt BIGINT,
conv_cnt BIGINT,
gmv_amount DOUBLE,
active_campaign_cnt BIGINT,
last_event_time TIMESTAMP,
ctr DOUBLE,
cvr DOUBLE
""".strip(),
        warehouse_dir / "dws" / "dws_user_daily",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dws.user_tag_snapshot",
        """
user_id STRING,
tags ARRAY<STRING>,
tag_cnt INT,
feature_7d_impressions BIGINT,
feature_7d_clicks BIGINT,
feature_7d_conversions BIGINT,
feature_7d_gmv DOUBLE,
feature_7d_ctr DOUBLE,
feature_7d_cvr DOUBLE,
feature_day_impressions BIGINT,
feature_day_clicks BIGINT,
feature_day_conversions BIGINT,
feature_day_gmv DOUBLE,
last_event_time TIMESTAMP
""".strip(),
        warehouse_dir / "dws" / "dws_user_tag_snapshot",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dws.tag_quality_daily",
        """
tag_name STRING,
tagged_user_cnt BIGINT,
total_user_cnt BIGINT,
coverage_rate DOUBLE,
avg_ctr_tagged DOUBLE,
avg_cvr_tagged DOUBLE,
avg_gmv_tagged DOUBLE
""".strip(),
        warehouse_dir / "dws" / "dws_tag_quality_daily",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "dws.tag_dict",
        """
tag_id STRING,
tag_name STRING,
tag_type STRING,
rule_desc STRING,
update_freq STRING,
create_time TIMESTAMP
""".strip(),
        warehouse_dir / "dws" / "dws_tag_dict",
        partitioned_by_dt=False,
    )

    # -------------------- ADS --------------------
    _create_external_parquet_table(
        spark,
        "ads.kpi_overview_daily",
        """
impressions BIGINT,
clicks BIGINT,
conversions BIGINT,
gmv DOUBLE,
cost DOUBLE,
ctr DOUBLE,
cvr DOUBLE,
rpm DOUBLE,
arpu DOUBLE,
active_users BIGINT,
roi DOUBLE
""".strip(),
        warehouse_dir / "ads" / "ads_kpi_overview_daily",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "ads.campaign_ranking_daily",
        """
campaign_id STRING,
impressions BIGINT,
clicks BIGINT,
conversions BIGINT,
gmv DOUBLE,
cost DOUBLE,
ctr DOUBLE,
cvr DOUBLE,
rpm DOUBLE,
roi DOUBLE,
rank_by_gmv INT,
rank_by_roi INT
""".strip(),
        warehouse_dir / "ads" / "ads_campaign_ranking_daily",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "ads.advertiser_dashboard_daily",
        """
advertiser_id STRING,
impressions BIGINT,
clicks BIGINT,
conversions BIGINT,
gmv DOUBLE,
cost DOUBLE,
ctr DOUBLE,
cvr DOUBLE,
roi DOUBLE,
revenue_growth_rate DOUBLE,
retention_flag INT
""".strip(),
        warehouse_dir / "ads" / "ads_advertiser_dashboard_daily",
        partitioned_by_dt=True,
    )
    _create_external_parquet_table(
        spark,
        "ads.tag_effectiveness_daily",
        """
tag_name STRING,
coverage_rate DOUBLE,
tagged_user_cnt BIGINT,
avg_ctr_tagged DOUBLE,
avg_cvr_tagged DOUBLE,
avg_gmv_tagged DOUBLE
""".strip(),
        warehouse_dir / "ads" / "ads_tag_effectiveness_daily",
        partitioned_by_dt=True,
    )

    spark.stop()


if __name__ == "__main__":
    main()

