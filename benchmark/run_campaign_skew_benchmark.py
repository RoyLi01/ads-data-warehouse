from __future__ import annotations

"""
Run a small reproducible benchmark for campaign_id aggregation skew.

The benchmark intentionally uses a simplified DWS-like task:

- read benchmark event data
- filter to valid impression rows
- aggregate by (dt, campaign_id)
- materialize the result

This keeps the experiment easy to run locally while still exposing the effect
of one hot campaign_id on shuffle/group-by runtime.

It now supports three logical modes:

- `uniform`: evenly distributed campaign_id aggregation
- `skewed`: one hot campaign_id dominates and is aggregated directly
- `skewed_salted`: same skewed input, but the hot campaign is salted and
  aggregated in two stages to mitigate skew
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import functions as F

from common.spark_session import build_spark


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _benchmark_root() -> Path:
    return _project_root() / "benchmark"


def _load_metadata(data_root: Path, distribution: str) -> Dict[str, Any]:
    meta_path = data_root / distribution / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing benchmark metadata: {meta_path}. "
            "Please run benchmark/generate_skew_data.py first."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _mode_distribution(mode: str) -> str:
    if mode == "uniform":
        return "uniform"
    if mode in {"skewed", "skewed_salted"}:
        return "skewed"
    raise ValueError(f"Unsupported mode: {mode}")


def _build_benchmark_spark(app_name: str, shuffle_partitions: int):
    tmp_root = _benchmark_root() / "tmp"
    hive_dir = tmp_root / "hive_warehouse"
    metastore_dir = tmp_root / "metastore_db"
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Reuse the shared Spark builder, but isolate benchmark metastore files so
    # the experiment does not interfere with the main project Hive metadata.
    #
    # AQE is disabled here on purpose: the benchmark wants to expose skew more
    # directly, while the main data-warehouse jobs reasonably keep AQE enabled.
    return build_spark(
        app_name=app_name,
        extra_conf={
            "spark.sql.adaptive.enabled": "false",
            "spark.sql.adaptive.coalescePartitions.enabled": "false",
            "spark.sql.adaptive.skewJoin.enabled": "false",
            "spark.sql.shuffle.partitions": str(shuffle_partitions),
            "spark.pyspark.driver.python": sys.executable,
            "spark.pyspark.python": sys.executable,
            "spark.sql.warehouse.dir": str(hive_dir),
            "spark.hadoop.hive.metastore.warehouse.dir": str(hive_dir),
            "spark.hadoop.javax.jdo.option.ConnectionURL": f"jdbc:derby:;databaseName={metastore_dir};create=true",
        },
    )


def _append_result_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name",
        "benchmark_mode",
        "mode",
        "distribution",
        "rows",
        "hot_ratio",
        "salt_buckets",
        "shuffle_partitions",
        "actual_output_rows",
        "partition_rows_min",
        "partition_rows_max",
        "partition_rows_avg",
        "partition_skew_ratio",
        "start_time",
        "end_time",
        "elapsed_sec",
        "timestamp",
    ]
    existing_rows: list[Dict[str, Any]] = []
    rewrite = False
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != fieldnames:
                rewrite = True
                for existing in reader:
                    existing_rows.append(existing)

    mode = "w" if rewrite or not csv_path.exists() else "a"
    with csv_path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow({k: existing.get(k) for k in fieldnames})
        writer.writerow({k: row.get(k) for k in fieldnames})


def _append_result_jsonl(jsonl_path: Path, row: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _partition_size_stats(df) -> Dict[str, float]:
    stats_row = (
        df.withColumn("_partition_id", F.spark_partition_id())
        .groupBy("_partition_id")
        .agg(F.count(F.lit(1)).alias("partition_rows"))
        .agg(
            F.min("partition_rows").alias("partition_rows_min"),
            F.max("partition_rows").alias("partition_rows_max"),
            F.avg("partition_rows").alias("partition_rows_avg"),
        )
        .collect()[0]
    )
    if stats_row["partition_rows_min"] is None:
        return {"partition_rows_min": 0.0, "partition_rows_max": 0.0, "partition_rows_avg": 0.0, "partition_skew_ratio": 0.0}
    min_rows = float(stats_row["partition_rows_min"])
    max_rows = float(stats_row["partition_rows_max"])
    avg_rows = float(stats_row["partition_rows_avg"])
    skew_ratio = float("inf") if avg_rows == 0 else round(max_rows / avg_rows, 2)
    return {
        "partition_rows_min": min_rows,
        "partition_rows_max": max_rows,
        "partition_rows_avg": round(avg_rows, 2),
        "partition_skew_ratio": skew_ratio,
    }


def _build_plain_stress_agg(filtered, shuffle_partitions: int):
    prepared = filtered.repartition(shuffle_partitions, F.col("campaign_id")).sortWithinPartitions("campaign_id", "event_time")
    agg = prepared.groupBy("dt", "campaign_id").agg(
        F.sum(F.when(F.col("event_type") == "impression", F.lit(1)).otherwise(F.lit(0))).alias("impressions"),
        F.sum(F.when(F.col("event_type") == "click", F.lit(1)).otherwise(F.lit(0))).alias("clicks"),
        F.count(F.lit(1)).alias("valid_events"),
        F.max("event_time").alias("last_event_time"),
    )
    return prepared, agg


def _build_salted_stress_agg(filtered, hot_campaign_id: str, salt_buckets: int, shuffle_partitions: int):
    # Salting only the hot campaign keeps the demo focused: the one overloaded
    # key is split into multiple sub-keys, then merged back in stage 2.
    salted = filtered.withColumn(
        "salt",
        F.when(
            F.col("campaign_id") == F.lit(hot_campaign_id),
            F.pmod(F.xxhash64("event_id"), F.lit(salt_buckets)),
        ).otherwise(F.lit(0)),
    )
    prepared = salted.repartition(shuffle_partitions, F.col("campaign_id"), F.col("salt")).sortWithinPartitions(
        "campaign_id", "salt", "event_time"
    )

    stage1 = prepared.groupBy("dt", "campaign_id", "salt").agg(
        F.sum(F.when(F.col("event_type") == "impression", F.lit(1)).otherwise(F.lit(0))).alias("impressions"),
        F.sum(F.when(F.col("event_type") == "click", F.lit(1)).otherwise(F.lit(0))).alias("clicks"),
        F.count(F.lit(1)).alias("valid_events"),
        F.max("event_time").alias("last_event_time"),
    )
    stage2 = stage1.groupBy("dt", "campaign_id").agg(
        F.sum("impressions").alias("impressions"),
        F.sum("clicks").alias("clicks"),
        F.sum("valid_events").alias("valid_events"),
        F.max("last_event_time").alias("last_event_time"),
    )
    return prepared, stage2


def _run_one_distribution(
    spark,
    *,
    data_root: Path,
    output_root: Path,
    mode: str,
    benchmark_mode: str,
    shuffle_partitions: int,
    salt_buckets: int,
) -> Dict[str, Any]:
    distribution = _mode_distribution(mode)
    meta = _load_metadata(data_root, distribution)
    input_path = data_root / distribution / "events"
    aggregate_output = output_root / mode / f"campaign_daily_{benchmark_mode}"

    start_dt = datetime.now()
    start_perf = time.perf_counter()

    df = spark.read.parquet(str(input_path))
    filtered = df.where(F.col("is_valid") == 1)

    if benchmark_mode == "simple":
        # Baseline: closest to the original minimal benchmark.
        agg = (
            filtered.where(F.col("event_type") == "impression")
            .groupBy("dt", "campaign_id")
            .agg(F.count(F.lit(1)).alias("impressions"))
        )
        partition_stats = {}
    elif benchmark_mode == "stress":
        # Stress mode makes key skew easier to observe locally. `skewed_salted`
        # further applies salting + two-stage aggregation to split the hot key
        # before merging it back to campaign_id grain.
        if mode == "skewed_salted":
            prepared, agg = _build_salted_stress_agg(
                filtered=filtered,
                hot_campaign_id=meta["hot_campaign_id"],
                salt_buckets=salt_buckets,
                shuffle_partitions=shuffle_partitions,
            )
        else:
            prepared, agg = _build_plain_stress_agg(filtered=filtered, shuffle_partitions=shuffle_partitions)
        partition_stats = _partition_size_stats(prepared)
    else:
        raise ValueError(f"Unsupported benchmark_mode: {benchmark_mode}")

    agg.write.mode("overwrite").parquet(str(aggregate_output))
    output_rows = agg.count()

    end_perf = time.perf_counter()
    end_dt = datetime.now()

    row = {
        "experiment_name": f"campaign_skew_benchmark_{benchmark_mode}",
        "benchmark_mode": benchmark_mode,
        "mode": mode,
        "distribution": distribution,
        "rows": int(meta["rows"]),
        "hot_ratio": float(meta["hot_ratio"]),
        "salt_buckets": int(salt_buckets if mode == "skewed_salted" else 0),
        "shuffle_partitions": int(shuffle_partitions),
        "actual_output_rows": int(output_rows),
        "start_time": start_dt.isoformat(timespec="seconds"),
        "end_time": end_dt.isoformat(timespec="seconds"),
        "elapsed_sec": round(end_perf - start_perf, 2),
        "timestamp": end_dt.isoformat(timespec="seconds"),
    }
    row.update(partition_stats)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DWS-like campaign aggregation under different skew profiles.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["uniform", "skewed", "skewed_salted", "all"],
        default=["all"],
        help="Logical experiment modes to run. `skewed_salted` reads skewed data but uses salting + two-stage aggregation.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Root directory containing generated benchmark data. Defaults to benchmark/data.",
    )
    parser.add_argument(
        "--result_root",
        default=None,
        help="Directory for benchmark outputs/results. Defaults to benchmark/results.",
    )
    parser.add_argument(
        "--benchmark_mode",
        choices=["simple", "stress"],
        default="stress",
        help="simple=plain groupBy; stress=repartition+sort+campaign aggregate to make skew easier to observe.",
    )
    parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=4,
        help="Fixed shuffle partition count used by the benchmark SparkSession.",
    )
    parser.add_argument(
        "--salt_buckets",
        type=int,
        default=8,
        help="Salt bucket count used only by skewed_salted mode.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else _benchmark_root() / "data"
    result_root = Path(args.result_root) if args.result_root else _benchmark_root() / "results"
    modes = ["uniform", "skewed", "skewed_salted"] if args.modes == ["all"] else args.modes

    spark = _build_benchmark_spark(app_name=f"campaign_skew_benchmark_{args.benchmark_mode}", shuffle_partitions=args.shuffle_partitions)
    results: list[Dict[str, Any]] = []
    try:
        for mode in modes:
            row = _run_one_distribution(
                spark,
                data_root=data_root,
                output_root=result_root,
                mode=mode,
                benchmark_mode=args.benchmark_mode,
                shuffle_partitions=args.shuffle_partitions,
                salt_buckets=args.salt_buckets,
            )
            results.append(row)
            _append_result_csv(result_root / "benchmark_results.csv", row)
            _append_result_jsonl(result_root / "benchmark_results.jsonl", row)
            print(
                f"experiment={row['experiment_name']} mode={row['mode']} distribution={row['distribution']} "
                f"rows={row['rows']} hot_ratio={row['hot_ratio']} salt_buckets={row['salt_buckets']} "
                f"elapsed_sec={row['elapsed_sec']}"
            )
            if "partition_skew_ratio" in row:
                print(
                    f"partition_rows_min={int(row['partition_rows_min'])} "
                    f"partition_rows_max={int(row['partition_rows_max'])} "
                    f"partition_rows_avg={row['partition_rows_avg']} "
                    f"partition_skew_ratio={row['partition_skew_ratio']}"
                )
    finally:
        spark.stop()

    result_map = {row["mode"]: row for row in results}
    if {"uniform", "skewed"}.issubset(result_map):
        uniform = result_map["uniform"]
        skewed = result_map["skewed"]
        slowdown = float("inf") if uniform["elapsed_sec"] == 0 else round(skewed["elapsed_sec"] / uniform["elapsed_sec"], 2)
        print(
            f"comparison=uniform_vs_skewed uniform_sec={uniform['elapsed_sec']} "
            f"skewed_sec={skewed['elapsed_sec']} slowdown_ratio={slowdown}"
        )
    if {"skewed", "skewed_salted"}.issubset(result_map):
        skewed = result_map["skewed"]
        salted = result_map["skewed_salted"]
        improvement = float("inf") if salted["elapsed_sec"] == 0 else round(skewed["elapsed_sec"] / salted["elapsed_sec"], 2)
        print(
            f"comparison=skewed_vs_skewed_salted skewed_sec={skewed['elapsed_sec']} "
            f"skewed_salted_sec={salted['elapsed_sec']} improvement_ratio={improvement}"
        )


if __name__ == "__main__":
    main()
