from __future__ import annotations

"""
Generate small benchmark event datasets for campaign skew experiments.

This module intentionally stays outside the main ODS/DWD/DWS pipeline. It
creates two independent datasets:

- `uniform`: campaign_id is close to evenly distributed
- `skewed`: one hot campaign_id takes most rows (default 80%)

The benchmark later reuses these event logs to observe how `GROUP BY dt,
campaign_id` behaves under different key distributions.
"""

import argparse
import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEVICES = ["ios", "android", "pc"]
USER_AGENTS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0)",
]
PAGES = ["home", "detail", "search", "feed"]
SITE_IDS = [f"site_{i:03d}" for i in range(1, 11)]
AD_SLOT_IDS = [f"slot_{i:03d}" for i in range(1, 21)]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("event_id", pa.string()),
            pa.field("event_time", pa.string()),
            pa.field("dt", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("ad_id", pa.string()),
            pa.field("campaign_id", pa.string()),
            pa.field("creative_id", pa.string()),
            pa.field("site_id", pa.string()),
            pa.field("page_id", pa.string()),
            pa.field("ad_slot_id", pa.string()),
            pa.field("device", pa.string()),
            pa.field("user_agent", pa.string()),
            pa.field("ip", pa.string()),
            pa.field("is_valid", pa.int32()),
            pa.field("event_type", pa.string()),
        ]
    )


def _campaign_ids(n_campaigns: int, hot_campaign_id: str) -> list[str]:
    others = [f"cmp_{i:04d}" for i in range(1, n_campaigns)]
    return [hot_campaign_id] + others


def _campaign_probabilities(campaign_ids: list[str], distribution: str, hot_ratio: float) -> np.ndarray:
    n = len(campaign_ids)
    if distribution == "uniform":
        return np.full(n, 1.0 / n)
    if distribution != "skewed":
        raise ValueError(f"Unsupported distribution: {distribution}")
    if not 0 < hot_ratio < 1:
        raise ValueError("hot_ratio must be between 0 and 1")
    if n < 2:
        return np.array([1.0])
    tail = (1.0 - hot_ratio) / (n - 1)
    return np.array([hot_ratio] + [tail] * (n - 1), dtype=float)


def _random_times_for_day(rng: np.random.Generator, dt: str, size: int) -> np.ndarray:
    base = datetime.strptime(dt, "%Y-%m-%d")
    secs = rng.integers(0, 86400, size=size, endpoint=False)
    return np.array([(base + timedelta(seconds=int(x))).strftime("%Y-%m-%d %H:%M:%S") for x in secs], dtype=object)


def _random_ips(rng: np.random.Generator, size: int) -> np.ndarray:
    octets = rng.integers(1, 255, size=(size, 4))
    return np.array([".".join(map(str, row)) for row in octets], dtype=object)


def _chunk_to_table(
    rng: np.random.Generator,
    *,
    distribution: str,
    dt: str,
    row_start: int,
    chunk_rows: int,
    campaign_ids: list[str],
    hot_ratio: float,
    click_rate: float,
    valid_rate: float,
    user_id_max: int,
) -> pa.Table:
    probs = _campaign_probabilities(campaign_ids, distribution, hot_ratio)
    seq = np.arange(row_start, row_start + chunk_rows, dtype=np.int64)
    campaign_idx = rng.choice(len(campaign_ids), size=chunk_rows, p=probs)
    campaign_values = np.take(np.array(campaign_ids, dtype=object), campaign_idx)

    user_numeric = rng.integers(1, user_id_max + 1, size=chunk_rows, endpoint=False)
    session_numeric = rng.integers(1, 500_000, size=chunk_rows, endpoint=False)
    ad_numeric = campaign_idx * 10 + rng.integers(0, 10, size=chunk_rows, endpoint=False)
    creative_numeric = campaign_idx * 100 + rng.integers(0, 20, size=chunk_rows, endpoint=False)

    event_types = np.where(rng.random(chunk_rows) < click_rate, "click", "impression")
    is_valid = np.where(rng.random(chunk_rows) < valid_rate, 1, 0).astype(np.int32)

    data = {
        "event_id": np.array([f"evt_{distribution}_{i:010d}" for i in seq], dtype=object),
        "event_time": _random_times_for_day(rng, dt, chunk_rows),
        "dt": np.full(chunk_rows, dt, dtype=object),
        "user_id": np.array([f"user_{x:07d}" for x in user_numeric], dtype=object),
        "session_id": np.array([f"sess_{x:08d}" for x in session_numeric], dtype=object),
        "ad_id": np.array([f"ad_{x:06d}" for x in ad_numeric], dtype=object),
        "campaign_id": campaign_values,
        "creative_id": np.array([f"crt_{x:07d}" for x in creative_numeric], dtype=object),
        "site_id": rng.choice(np.array(SITE_IDS, dtype=object), size=chunk_rows),
        "page_id": rng.choice(np.array(PAGES, dtype=object), size=chunk_rows),
        "ad_slot_id": rng.choice(np.array(AD_SLOT_IDS, dtype=object), size=chunk_rows),
        "device": rng.choice(np.array(DEVICES, dtype=object), size=chunk_rows),
        "user_agent": rng.choice(np.array(USER_AGENTS, dtype=object), size=chunk_rows),
        "ip": _random_ips(rng, chunk_rows),
        "is_valid": is_valid,
        "event_type": event_types.astype(object),
    }
    return pa.Table.from_pydict(data, schema=_build_schema())


def _write_distribution(
    *,
    output_root: Path,
    distribution: str,
    rows: int,
    dt: str,
    n_campaigns: int,
    hot_ratio: float,
    click_rate: float,
    valid_rate: float,
    chunk_size: int,
    seed: int,
) -> Dict[str, object]:
    dist_dir = output_root / distribution
    events_dir = dist_dir / "events"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    events_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    hot_campaign_id = "cmp_hot_0001"
    campaign_ids = _campaign_ids(n_campaigns=n_campaigns, hot_campaign_id=hot_campaign_id)

    writer: pq.ParquetWriter | None = None
    written = 0
    file_idx = 0

    # Chunked generation keeps memory stable even when default rows=3,000,000.
    while written < rows:
        current = min(chunk_size, rows - written)
        table = _chunk_to_table(
            rng,
            distribution=distribution,
            dt=dt,
            row_start=written,
            chunk_rows=current,
            campaign_ids=campaign_ids,
            hot_ratio=hot_ratio,
            click_rate=click_rate,
            valid_rate=valid_rate,
            user_id_max=max(200_000, rows // 6),
        )
        file_path = events_dir / f"part-{file_idx:05d}.parquet"
        pq.write_table(table, file_path, compression="snappy")
        written += current
        file_idx += 1

    actual_probs = _campaign_probabilities(campaign_ids, distribution, hot_ratio)
    meta = {
        "distribution": distribution,
        "rows": rows,
        "dt": dt,
        "n_campaigns": n_campaigns,
        "hot_campaign_id": hot_campaign_id,
        "hot_ratio": hot_ratio if distribution == "skewed" else 1.0 / n_campaigns,
        "target_hot_ratio": hot_ratio,
        "click_rate": click_rate,
        "valid_rate": valid_rate,
        "chunk_size": chunk_size,
        "seed": seed,
        "top_campaign_expected_share": float(actual_probs[0]),
        "events_path": str(events_dir),
    }
    (dist_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark event data for campaign skew experiments.")
    parser.add_argument("--rows", type=int, default=3_000_000, help="Rows per distribution dataset.")
    parser.add_argument("--dt", default="2026-03-07", help="Single benchmark dt.")
    parser.add_argument("--campaigns", type=int, default=200, help="Number of distinct campaign_id values.")
    parser.add_argument("--hot_ratio", type=float, default=0.8, help="Hot campaign share in skewed distribution.")
    parser.add_argument("--click_rate", type=float, default=0.03, help="Share of rows labeled as click.")
    parser.add_argument("--valid_rate", type=float, default=0.98, help="Share of rows marked as valid.")
    parser.add_argument("--chunk_size", type=int, default=250_000, help="Rows written per parquet chunk.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--output_root",
        default=None,
        help="Root directory for generated benchmark data. Defaults to benchmark/data.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else _project_root() / "benchmark" / "data"
    output_root.mkdir(parents=True, exist_ok=True)

    metas = []
    for idx, distribution in enumerate(["uniform", "skewed"]):
        meta = _write_distribution(
            output_root=output_root,
            distribution=distribution,
            rows=args.rows,
            dt=args.dt,
            n_campaigns=args.campaigns,
            hot_ratio=args.hot_ratio,
            click_rate=args.click_rate,
            valid_rate=args.valid_rate,
            chunk_size=args.chunk_size,
            seed=args.seed + idx,
        )
        metas.append(meta)

    print("=== benchmark skew data generated ===")
    for meta in metas:
        print(
            f"distribution={meta['distribution']} rows={meta['rows']} dt={meta['dt']} "
            f"campaigns={meta['n_campaigns']} expected_top_share={meta['top_campaign_expected_share']:.4f} "
            f"path={meta['events_path']}"
        )


if __name__ == "__main__":
    main()
