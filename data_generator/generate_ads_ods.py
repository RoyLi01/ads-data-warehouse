from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CampaignPricing:
    cpm: float  # cost per 1000 impressions
    cpc: float  # cost per click


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fmt_dt(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _rand_times_within_day(rng: np.random.Generator, dt: str, n: int) -> np.ndarray:
    base = _parse_dt(dt)
    secs = rng.integers(0, 86400, size=n, endpoint=False)
    ts = [base + timedelta(seconds=int(x)) for x in secs]
    return np.array([t.strftime("%Y-%m-%d %H:%M:%S") for t in ts], dtype=object)


def _add_seconds(ts_str: str, seconds: int) -> str:
    t = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=seconds)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _add_minutes(ts_str: str, minutes: int) -> str:
    t = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=minutes)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _make_campaign_ids(n_other: int, hot_campaign_id: str) -> List[str]:
    others = [f"cmp_{i:04d}" for i in range(1, n_other + 1)]
    return [hot_campaign_id] + others


def _campaign_ctr_map(rng: np.random.Generator, campaign_ids: Iterable[str], ctr_min: float, ctr_max: float) -> Dict[str, float]:
    ids = list(campaign_ids)
    ctrs = rng.uniform(ctr_min, ctr_max, size=len(ids))
    return {cid: float(p) for cid, p in zip(ids, ctrs)}


def _campaign_cvr_map(rng: np.random.Generator, campaign_ids: Iterable[str], cvr_min: float, cvr_max: float) -> Dict[str, float]:
    ids = list(campaign_ids)
    cvrs = rng.uniform(cvr_min, cvr_max, size=len(ids))
    return {cid: float(p) for cid, p in zip(ids, cvrs)}


def _campaign_pricing_map(rng: np.random.Generator, campaign_ids: Iterable[str]) -> Dict[str, CampaignPricing]:
    mp: Dict[str, CampaignPricing] = {}
    for cid in campaign_ids:
        cpm = float(rng.uniform(10.0, 60.0))  # RMB per 1000 impressions
        cpc = float(rng.uniform(0.2, 2.0))    # RMB per click
        mp[cid] = CampaignPricing(cpm=cpm, cpc=cpc)
    return mp


def _allocate_impressions(
    rng: np.random.Generator,
    total_imps: int,
    campaign_ids: List[str],
    hot_campaign_id: str,
    hot_share: float,
) -> List[Tuple[str, int]]:
    if total_imps <= 0:
        return []
    if hot_campaign_id not in campaign_ids:
        raise ValueError("hot_campaign_id must be included in campaign_ids")

    hot_imps = int(round(total_imps * hot_share))
    hot_imps = max(1, min(total_imps, hot_imps))
    rest = total_imps - hot_imps

    others = [c for c in campaign_ids if c != hot_campaign_id]
    if not others:
        return [(hot_campaign_id, total_imps)]

    # distribute remaining impressions across other campaigns (small + random)
    weights = rng.random(len(others))
    weights = weights / weights.sum()
    alloc = (weights * rest).astype(int)
    # fix rounding drift
    drift = rest - int(alloc.sum())
    if drift != 0:
        idx = rng.integers(0, len(alloc), size=abs(drift))
        for i in idx:
            alloc[i] += 1 if drift > 0 else -1
    alloc = np.maximum(0, alloc)

    out = [(hot_campaign_id, hot_imps)]
    out.extend(zip(others, alloc.tolist()))
    return out


def _gen_event_log_for_day(
    rng: np.random.Generator,
    dt: str,
    campaign_ids: List[str],
    hot_campaign_id: str,
    total_imps: int,
    hot_share: float,
    ctr_map: Dict[str, float],
    devices: List[str],
    site_ids: List[str],
    ad_id_pool: List[str],
    creative_id_pool: List[str],
    user_id_max: int,
    starting_event_seq: int,
) -> Tuple[pd.DataFrame, int]:
    alloc = _allocate_impressions(rng, total_imps=total_imps, campaign_ids=campaign_ids, hot_campaign_id=hot_campaign_id, hot_share=hot_share)

    # Build impression rows campaign by campaign (keeps hot campaign skew obvious)
    imp_parts: List[pd.DataFrame] = []
    seq = starting_event_seq

    for campaign_id, n_imps in alloc:
        if n_imps <= 0:
            continue
        event_ids = [f"evt_imp_{dt.replace('-', '')}_{seq + i:010d}" for i in range(n_imps)]
        seq += n_imps

        imp_df = pd.DataFrame(
            {
                "event_id": event_ids,
                "event_time": _rand_times_within_day(rng, dt=dt, n=n_imps),
                "dt": dt,
                "user_id": rng.integers(1, user_id_max + 1, size=n_imps).astype(str),
                "ad_id": rng.choice(ad_id_pool, size=n_imps),
                "campaign_id": campaign_id,
                "creative_id": rng.choice(creative_id_pool, size=n_imps),
                "site_id": rng.choice(site_ids, size=n_imps),
                "event_type": "impression",
                "device": rng.choice(devices, size=n_imps),
            }
        )
        imp_parts.append(imp_df)

    impressions = pd.concat(imp_parts, ignore_index=True) if imp_parts else pd.DataFrame(
        columns=["event_id", "event_time", "dt", "user_id", "ad_id", "campaign_id", "creative_id", "site_id", "event_type", "device"]
    )

    # Sample clicks from impressions using campaign-level CTR (overall ~1%-5%)
    if len(impressions) > 0:
        p = impressions["campaign_id"].map(ctr_map).astype(float).to_numpy()
        click_mask = rng.random(len(impressions)) < p
        clicked = impressions.loc[click_mask].copy()
    else:
        clicked = impressions.copy()

    if len(clicked) > 0:
        clk_ids = [f"evt_clk_{dt.replace('-', '')}_{seq + i:010d}" for i in range(len(clicked))]
        seq += len(clicked)
        clicked["event_id"] = clk_ids
        # click happens shortly after impression
        add_secs = rng.integers(1, 301, size=len(clicked), endpoint=False)
        clicked["event_time"] = [
            _add_seconds(t, int(s)) for t, s in zip(clicked["event_time"].tolist(), add_secs.tolist())
        ]
        clicked["event_type"] = "click"

    event_log = pd.concat([impressions, clicked], ignore_index=True)
    return event_log, seq


def _gen_conversions(
    rng: np.random.Generator,
    event_log: pd.DataFrame,
    cvr_map: Dict[str, float],
    starting_conv_seq: int,
    max_delay_minutes: int,
) -> Tuple[pd.DataFrame, int]:
    clicks = event_log[event_log["event_type"] == "click"].copy()
    if len(clicks) == 0:
        cols = ["conv_id", "conv_time", "dt", "user_id", "ad_id", "campaign_id", "order_id", "gmv_amount"]
        return pd.DataFrame(columns=cols), starting_conv_seq

    p = clicks["campaign_id"].map(cvr_map).astype(float).to_numpy()
    conv_mask = rng.random(len(clicks)) < p
    conv_clicks = clicks.loc[conv_mask].copy()

    if len(conv_clicks) == 0:
        cols = ["conv_id", "conv_time", "dt", "user_id", "ad_id", "campaign_id", "order_id", "gmv_amount"]
        return pd.DataFrame(columns=cols), starting_conv_seq

    conv_ids = [f"conv_{c.replace('-', '')}_{starting_conv_seq + i:010d}" for i, c in enumerate(conv_clicks["dt"].tolist())]
    seq = starting_conv_seq + len(conv_clicks)

    delays = rng.integers(1, max_delay_minutes + 1, size=len(conv_clicks), endpoint=True)
    conv_times = [_add_minutes(t, int(m)) for t, m in zip(conv_clicks["event_time"].tolist(), delays.tolist())]

    # gmv: right-skewed distribution
    gmv = rng.lognormal(mean=math.log(80), sigma=0.7, size=len(conv_clicks))
    gmv = np.clip(gmv, 5, 5000)

    order_ids = [f"ord_{starting_conv_seq + i:012d}" for i in range(len(conv_clicks))]

    conv_df = pd.DataFrame(
        {
            "conv_id": conv_ids,
            "conv_time": conv_times,
            "dt": conv_clicks["dt"].tolist(),
            "user_id": conv_clicks["user_id"].tolist(),
            "ad_id": conv_clicks["ad_id"].tolist(),
            "campaign_id": conv_clicks["campaign_id"].tolist(),
            "order_id": order_ids,
            "gmv_amount": np.round(gmv, 2),
        }
    )
    return conv_df, seq


def _gen_cost(
    rng: np.random.Generator,
    event_log: pd.DataFrame,
    pricing_map: Dict[str, CampaignPricing],
) -> pd.DataFrame:
    imps = event_log[event_log["event_type"] == "impression"].groupby(["dt", "campaign_id"], as_index=False).size()
    clks = event_log[event_log["event_type"] == "click"].groupby(["dt", "campaign_id"], as_index=False).size()

    merged = imps.merge(clks, on=["dt", "campaign_id"], how="outer", suffixes=("_imps", "_clks")).fillna(0)
    merged = merged.rename(columns={"size_imps": "impressions", "size_clks": "clicks"})

    def _row_cost(row) -> float:
        pricing = pricing_map.get(row["campaign_id"])
        if pricing is None:
            pricing = CampaignPricing(cpm=30.0, cpc=1.0)
        base = (float(row["impressions"]) * pricing.cpm / 1000.0) + (float(row["clicks"]) * pricing.cpc)
        noise = float(rng.uniform(0.98, 1.02))
        return max(0.0, base * noise)

    merged["cost"] = merged.apply(_row_cost, axis=1)
    merged["cost"] = merged["cost"].round(4)

    return merged[["dt", "campaign_id", "cost"]].sort_values(["dt", "campaign_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock ad ODS CSVs into data/ods/")
    parser.add_argument("--start_dt", default="2026-03-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=3, help="Number of days to generate (>=3 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", default=str(_project_root() / "data" / "ods"), help="Output directory")

    parser.add_argument("--n_other_campaigns", type=int, default=200, help="Number of non-hot campaigns")
    parser.add_argument("--hot_campaign_id", default="cmp_hot_0001", help="Hot campaign id for skew demo")
    parser.add_argument("--daily_impressions", type=int, default=80000, help="Total impressions per day")
    parser.add_argument("--hot_share", type=float, default=0.7, help="Share of impressions assigned to hot campaign")

    parser.add_argument("--ctr_min", type=float, default=0.01, help="Min CTR per campaign")
    parser.add_argument("--ctr_max", type=float, default=0.05, help="Max CTR per campaign")
    parser.add_argument("--cvr_min", type=float, default=0.02, help="Min click->conv rate per campaign")
    parser.add_argument("--cvr_max", type=float, default=0.10, help="Max click->conv rate per campaign")
    args = parser.parse_args()

    if args.days < 1:
        raise ValueError("--days must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    campaign_ids = _make_campaign_ids(n_other=args.n_other_campaigns, hot_campaign_id=args.hot_campaign_id)
    ctr_map = _campaign_ctr_map(rng, campaign_ids, ctr_min=args.ctr_min, ctr_max=args.ctr_max)
    cvr_map = _campaign_cvr_map(rng, campaign_ids, cvr_min=args.cvr_min, cvr_max=args.cvr_max)
    pricing_map = _campaign_pricing_map(rng, campaign_ids)

    devices = ["ios", "android", "windows", "macos"]
    site_ids = [f"site_{i:03d}" for i in range(1, 51)]
    ad_id_pool = [f"ad_{i:06d}" for i in range(1, 2001)]
    creative_id_pool = [f"cr_{i:06d}" for i in range(1, 5001)]

    event_parts: List[pd.DataFrame] = []
    conv_parts: List[pd.DataFrame] = []

    start = _parse_dt(args.start_dt)
    event_seq = 1
    conv_seq = 1

    for i in range(args.days):
        dt = _fmt_dt(start + timedelta(days=i))
        event_log_day, event_seq = _gen_event_log_for_day(
            rng=rng,
            dt=dt,
            campaign_ids=campaign_ids,
            hot_campaign_id=args.hot_campaign_id,
            total_imps=args.daily_impressions,
            hot_share=args.hot_share,
            ctr_map=ctr_map,
            devices=devices,
            site_ids=site_ids,
            ad_id_pool=ad_id_pool,
            creative_id_pool=creative_id_pool,
            user_id_max=500_000,
            starting_event_seq=event_seq,
        )
        event_parts.append(event_log_day)

    event_log = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()

    # conversions are sampled from clicks
    conv_df, conv_seq = _gen_conversions(
        rng=rng,
        event_log=event_log,
        cvr_map=cvr_map,
        starting_conv_seq=conv_seq,
        max_delay_minutes=120,
    )
    conv_parts.append(conv_df)
    conversion_log = pd.concat(conv_parts, ignore_index=True) if conv_parts else pd.DataFrame()

    cost_df = _gen_cost(rng=rng, event_log=event_log, pricing_map=pricing_map)

    # write outputs
    event_out = out_dir / "ods_ad_event_log.csv"
    conv_out = out_dir / "ods_conversion_log.csv"
    cost_out = out_dir / "ods_ad_cost.csv"

    # ensure column order exactly as required
    event_cols = [
        "event_id",
        "event_time",
        "dt",
        "user_id",
        "ad_id",
        "campaign_id",
        "creative_id",
        "site_id",
        "event_type",
        "device",
    ]
    conv_cols = ["conv_id", "conv_time", "dt", "user_id", "ad_id", "campaign_id", "order_id", "gmv_amount"]
    cost_cols = ["dt", "campaign_id", "cost"]

    event_log[event_cols].to_csv(event_out, index=False)
    conversion_log[conv_cols].to_csv(conv_out, index=False)
    cost_df[cost_cols].to_csv(cost_out, index=False)

    # quick stats for sanity
    imps = int((event_log["event_type"] == "impression").sum()) if len(event_log) else 0
    clks = int((event_log["event_type"] == "click").sum()) if len(event_log) else 0
    ctr = (clks / imps) if imps else 0.0
    convs = len(conversion_log)
    cvr = (convs / clks) if clks else 0.0
    hot_imps = int(((event_log["campaign_id"] == args.hot_campaign_id) & (event_log["event_type"] == "impression")).sum())
    hot_share_real = (hot_imps / imps) if imps else 0.0

    print(f"Output dir: {out_dir}")
    print(f"- ods_ad_event_log.csv: {len(event_log):,} rows (imps={imps:,}, clicks={clks:,}, CTR={ctr:.2%})")
    print(f"- ods_conversion_log.csv: {convs:,} rows (click->conv={cvr:.2%})")
    print(f"- ods_ad_cost.csv: {len(cost_df):,} rows")
    print(f"Hot campaign: {args.hot_campaign_id} impressions share={hot_share_real:.2%}")


if __name__ == "__main__":
    main()

