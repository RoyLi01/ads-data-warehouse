from __future__ import annotations

"""
Generate mock ODS CSVs (tracking-log-like + dims/config tables).

This is the single official ODS generator of the project.

Main outputs (default under `data/ods/`):

- `ods_ad_event_log.csv` (+ `dt=.../ods_ad_event_log.csv`)
  - **Grain**: 1 row per event (impression / click)
  - **How**:
    - Generate impressions with a hot campaign (skew)
    - Derive clicks by sampling impressions (per-campaign CTR)
    - Derive session_id by sessionization within (dt, user_id) with 30-min gap rule
    - Derive ad_slot_id/page_id/user_agent/ip as random enrichments
    - Derive is_valid with different invalid rates for click vs impression

- `ods_conversion_log.csv` (+ `dt=.../ods_conversion_log.csv`)
  - **Grain**: 1 row per conversion (treated as 1 purchase/order)
  - **How**: sample from clicks (per-campaign CVR), conv_time = click_time + delay, GMV ~ lognormal

- `ods_ad_cost.csv` (+ `dt=.../ods_ad_cost.csv`)
  - **Grain**: 1 row per (dt, campaign_id)
  - **How**: simple CPM+CPC pricing + small noise

- `ods_user_profile.csv`
  - **Grain**: 1 row per user_id (only users that appear in event_log)

- `ods_ad_meta.csv`
  - **Grain**: 1 row per (ad_id, campaign_id) pair appearing in event_log
  - **Includes**: advertiser_id/ad_type/landing_type/product_id/start_dt/end_dt

- `ods_ad_slot.csv`
  - **Grain**: 1 row per ad_slot_id (slot_001..slot_020)
  - **Includes**: slot_type/app/position/price_factor
"""

import argparse
import hashlib
import math
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


REGIONS = ["北京", "上海", "广东", "浙江", "江苏", "四川", "湖北"]
PAGES = ["home", "detail", "search", "live"]
SLOT_IDS = [f"slot_{i:03d}" for i in range(1, 21)]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fmt_dt(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _stable_int(s: str) -> int:
    # Stable hash across runs/platforms (avoid Python's salted hash()).
    # Used for mapping campaign_id -> advertiser_id deterministically.
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


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
        cpm = float(rng.uniform(10.0, 60.0))
        cpc = float(rng.uniform(0.2, 2.0))
        mp[cid] = CampaignPricing(cpm=cpm, cpc=cpc)
    return mp


def _allocate_impressions(
    rng: np.random.Generator,
    total_imps: int,
    campaign_ids: List[str],
    hot_campaign_id: str,
    hot_share: float,
) -> List[Tuple[str, int]]:
    if hot_campaign_id not in campaign_ids:
        raise ValueError("hot_campaign_id must be included in campaign_ids")
    hot_imps = int(round(total_imps * hot_share))
    hot_imps = min(max(hot_imps, 0), total_imps)
    rest = int(total_imps - hot_imps)

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

    # gmv: right-skewed distribution (more realistic than uniform).
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


def _device_type_from_device(device: str) -> str:
    if device == "ios":
        return "ios"
    if device == "android":
        return "android"
    return "web"


def _user_agent_from_device_type(device_type: str) -> str:
    if device_type == "ios":
        return "ios_app"
    if device_type == "android":
        return "android_app"
    return "web"


def _gen_ipv4(rng: np.random.Generator, n: int) -> pd.Series:
    # "Looks like IPv4" is enough for demo; no need to model real distributions.
    a = rng.integers(1, 223, size=n)
    b = rng.integers(0, 255, size=n)
    c = rng.integers(0, 255, size=n)
    d = rng.integers(1, 255, size=n)
    return pd.Series([f"{x}.{y}.{z}.{w}" for x, y, z, w in zip(a, b, c, d)], dtype="string")


def _add_session_id(event_log: pd.DataFrame) -> pd.Series:
    # Sessionization rule:
    # within the same (dt, user_id), sort by event_time; if gap > 30 minutes => start a new session.
    tmp = event_log[["dt", "user_id", "event_time"]].copy()
    tmp["event_time_ts"] = pd.to_datetime(tmp["event_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    tmp = tmp.sort_values(["dt", "user_id", "event_time_ts"], kind="mergesort")

    diffs = tmp.groupby(["dt", "user_id"])["event_time_ts"].diff()
    new_sess = diffs.isna() | (diffs > pd.Timedelta(minutes=30))
    sess_seq = new_sess.groupby([tmp["dt"], tmp["user_id"]]).cumsum()
    sess_seq = sess_seq.astype(int) + 1

    dt_compact = tmp["dt"].str.replace("-", "", regex=False)
    session_id = "sess_" + dt_compact + "_" + tmp["user_id"].astype(str) + "_" + sess_seq.astype(str).str.zfill(3)

    # restore to original row order
    session_id = session_id.reindex(event_log.index)
    return session_id


def _gen_user_profile(rng: np.random.Generator, event_log: pd.DataFrame, start_dt: str) -> pd.DataFrame:
    # Only generate profiles for users that actually appear in the event log.
    users = event_log[["user_id", "device"]].copy()
    users["device_type"] = users["device"].map(_device_type_from_device)

    # choose most frequent device_type per user (fallback to first)
    device_pref = (
        users.groupby("user_id")["device_type"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        .reset_index()
    )

    user_ids = device_pref["user_id"].astype(str).tolist()
    n = len(user_ids)

    gender = rng.choice(["M", "F", "U"], size=n, p=[0.48, 0.48, 0.04])
    age = rng.integers(16, 61, size=n)
    region = rng.choice(REGIONS, size=n)

    start = _parse_dt(start_dt)
    reg_offsets = rng.integers(0, 365, size=n)
    register_dt = [(start - timedelta(days=int(x))).strftime("%Y-%m-%d") for x in reg_offsets]

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "gender": gender,
            "age": age.astype(int),
            "region": region,
            "device_type": device_pref["device_type"].tolist(),
            "register_dt": register_dt,
        }
    )


def _gen_ad_meta(rng: np.random.Generator, event_log: pd.DataFrame, start_dt: str, end_dt: str) -> pd.DataFrame:
    # Use (ad_id, campaign_id) as the grain (closer to real-world: the same ad_id can be reused).
    pairs = event_log[["ad_id", "campaign_id"]].drop_duplicates().reset_index(drop=True)
    n = len(pairs)

    advertiser_n = 50
    advertiser_id = [
        f"adv_{(_stable_int(cid) % advertiser_n) + 1:03d}" for cid in pairs["campaign_id"].astype(str).tolist()
    ]
    ad_type = rng.choice(["video", "banner"], size=n, p=[0.6, 0.4])
    landing_type = rng.choice(["app", "web"], size=n, p=[0.7, 0.3])

    # product_id (long-tail distribution: hot products + tail products)
    # - total products: 5000
    # - top 5% (250) are "hot": product_0001 ~ product_0250, chosen with 60% probability
    # - the rest are "tail": product_0251 ~ product_5000, chosen with 40% probability
    # - keep 20% ads without product (empty string) for realism
    has_product = rng.random(n) >= 0.2
    pick_hot = rng.random(n) < 0.6

    product_id_arr = np.full(n, "", dtype=object)
    hot_mask = has_product & pick_hot
    tail_mask = has_product & (~pick_hot)

    if int(hot_mask.sum()) > 0:
        hot_nums = rng.integers(1, 251, size=int(hot_mask.sum()))
        product_id_arr[hot_mask] = [f"product_{x:04d}" for x in hot_nums.tolist()]
    if int(tail_mask.sum()) > 0:
        tail_nums = rng.integers(251, 5001, size=int(tail_mask.sum()))
        product_id_arr[tail_mask] = [f"product_{x:04d}" for x in tail_nums.tolist()]

    product_id = product_id_arr.tolist()

    # Make the campaign effective window cover the generation range (with some jitter)
    start_base = _parse_dt(start_dt)
    end_base = _parse_dt(end_dt)
    start_jitter = rng.integers(0, 8, size=n)
    end_jitter = rng.integers(0, 8, size=n)
    start_dt_col = [(start_base - timedelta(days=int(x))).strftime("%Y-%m-%d") for x in start_jitter]
    end_dt_col = [(end_base + timedelta(days=int(x))).strftime("%Y-%m-%d") for x in end_jitter]

    return pd.DataFrame(
        {
            "ad_id": pairs["ad_id"].astype(str),
            "campaign_id": pairs["campaign_id"].astype(str),
            "advertiser_id": advertiser_id,
            "ad_type": ad_type,
            "landing_type": landing_type,
            "product_id": product_id,
            "start_dt": start_dt_col,
            "end_dt": end_dt_col,
        }
    )


def _gen_ad_slot(rng: np.random.Generator) -> pd.DataFrame:
    slot_type = rng.choice(["feed", "search", "live", "detail"], size=len(SLOT_IDS))
    app = rng.choice(["main_app", "lite_app", "web"], size=len(SLOT_IDS), p=[0.6, 0.25, 0.15])
    position = rng.choice(["top", "middle", "bottom"], size=len(SLOT_IDS), p=[0.2, 0.6, 0.2])
    price_factor = np.round(rng.uniform(0.8, 1.2, size=len(SLOT_IDS)), 3)

    return pd.DataFrame(
        {
            "ad_slot_id": SLOT_IDS,
            "slot_type": slot_type,
            "app": app,
            "position": position,
            "price_factor": price_factor,
        }
    )


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enhanced ODS CSVs into data/ods/")
    parser.add_argument("--start_dt", default="2026-03-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=3, help="Number of days to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", default=str(_project_root() / "data" / "ods"), help="Output directory")
    args = parser.parse_args()

    if args.days < 1:
        raise ValueError("--days must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Keep the same behavior distribution as the current minimal project:
    # - 200 non-hot campaigns + 1 hot campaign (for skew demo)
    # - CTR ~ 1%-5%, click->conv ~ 2%-10%
    n_other_campaigns = 200
    hot_campaign_id = "cmp_hot_0001"
    daily_impressions = 80_000
    hot_share = 0.7
    ctr_min, ctr_max = 0.01, 0.05
    cvr_min, cvr_max = 0.02, 0.10

    campaign_ids = _make_campaign_ids(n_other=n_other_campaigns, hot_campaign_id=hot_campaign_id)
    ctr_map = _campaign_ctr_map(rng, campaign_ids, ctr_min=ctr_min, ctr_max=ctr_max)
    cvr_map = _campaign_cvr_map(rng, campaign_ids, cvr_min=cvr_min, cvr_max=cvr_max)
    pricing_map = _campaign_pricing_map(rng, campaign_ids)

    devices = ["ios", "android", "windows", "macos"]
    site_ids = [f"site_{i:03d}" for i in range(1, 51)]
    ad_id_pool = [f"ad_{i:06d}" for i in range(1, 2001)]
    creative_id_pool = [f"cr_{i:06d}" for i in range(1, 5001)]

    start = _parse_dt(args.start_dt)
    end = start + timedelta(days=args.days - 1)
    end_dt_str = _fmt_dt(end)

    event_parts: List[pd.DataFrame] = []
    event_seq = 1
    for i in range(args.days):
        dt = _fmt_dt(start + timedelta(days=i))
        event_log_day, event_seq = _gen_event_log_for_day(
            rng=rng,
            dt=dt,
            campaign_ids=campaign_ids,
            hot_campaign_id=hot_campaign_id,
            total_imps=daily_impressions,
            hot_share=hot_share,
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
    conv_log, _ = _gen_conversions(rng=rng, event_log=event_log, cvr_map=cvr_map, starting_conv_seq=1, max_delay_minutes=120)
    cost_df = _gen_cost(rng=rng, event_log=event_log, pricing_map=pricing_map)

    # ===================== Embedded-like fields (append to CSV tail) =====================
    session_id = _add_session_id(event_log)
    ad_slot_id = rng.choice(SLOT_IDS, size=len(event_log))
    page_id = rng.choice(PAGES, size=len(event_log))

    # invalid simulation: impression 1% invalid, click 2% invalid
    invalid_rate = np.where(event_log["event_type"].to_numpy() == "click", 0.02, 0.01)
    is_valid = (rng.random(len(event_log)) >= invalid_rate).astype(int)

    ip = _gen_ipv4(rng, len(event_log))

    device_type = event_log["device"].astype(str).map(_device_type_from_device)
    user_agent = device_type.map(_user_agent_from_device_type)

    event_log_enriched = event_log.copy()
    event_log_enriched["session_id"] = session_id.astype(str)
    event_log_enriched["ad_slot_id"] = ad_slot_id
    event_log_enriched["page_id"] = page_id
    event_log_enriched["is_valid"] = is_valid
    event_log_enriched["ip"] = ip
    event_log_enriched["user_agent"] = user_agent

    # ===================== Dimension/config tables =====================
    ods_user_profile = _gen_user_profile(rng, event_log_enriched, start_dt=args.start_dt)
    ods_ad_meta = _gen_ad_meta(rng, event_log_enriched, start_dt=args.start_dt, end_dt=end_dt_str)
    ods_ad_slot = _gen_ad_slot(rng)

    # ===================== Write outputs (compat + dt partition dirs) =====================
    event_cols_v1 = [
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
    event_cols_tail = ["session_id", "ad_slot_id", "page_id", "is_valid", "ip", "user_agent"]
    event_cols = event_cols_v1 + event_cols_tail

    conv_cols = ["conv_id", "conv_time", "dt", "user_id", "ad_id", "campaign_id", "order_id", "gmv_amount"]
    cost_cols = ["dt", "campaign_id", "cost"]

    paths_written: List[Tuple[str, int]] = []

    p_event_root = out_dir / "ods_ad_event_log.csv"
    p_conv_root = out_dir / "ods_conversion_log.csv"
    p_cost_root = out_dir / "ods_ad_cost.csv"
    _write_csv(event_log_enriched[event_cols], p_event_root)
    _write_csv(conv_log[conv_cols], p_conv_root)
    _write_csv(cost_df[cost_cols], p_cost_root)
    paths_written.extend(
        [
            (str(p_event_root), len(event_log_enriched)),
            (str(p_conv_root), len(conv_log)),
            (str(p_cost_root), len(cost_df)),
        ]
    )

    # Additional dt partition directory outputs
    for dt, df_day in event_log_enriched.groupby("dt", sort=True):
        dt_dir = out_dir / f"dt={dt}"
        _write_csv(df_day[event_cols], dt_dir / "ods_ad_event_log.csv")
        paths_written.append((str(dt_dir / "ods_ad_event_log.csv"), len(df_day)))

        conv_day = conv_log[conv_log["dt"] == dt]
        _write_csv(conv_day[conv_cols], dt_dir / "ods_conversion_log.csv")
        paths_written.append((str(dt_dir / "ods_conversion_log.csv"), len(conv_day)))

        cost_day = cost_df[cost_df["dt"] == dt]
        _write_csv(cost_day[cost_cols], dt_dir / "ods_ad_cost.csv")
        paths_written.append((str(dt_dir / "ods_ad_cost.csv"), len(cost_day)))

    # New dimension/config CSVs (root only)
    p_user = out_dir / "ods_user_profile.csv"
    p_meta = out_dir / "ods_ad_meta.csv"
    p_slot = out_dir / "ods_ad_slot.csv"
    _write_csv(ods_user_profile, p_user)
    _write_csv(ods_ad_meta, p_meta)
    _write_csv(ods_ad_slot, p_slot)
    paths_written.extend([(str(p_user), len(ods_user_profile)), (str(p_meta), len(ods_ad_meta)), (str(p_slot), len(ods_ad_slot))])

    # ===================== Summary =====================
    imps = int((event_log_enriched["event_type"] == "impression").sum()) if len(event_log_enriched) else 0
    clks = int((event_log_enriched["event_type"] == "click").sum()) if len(event_log_enriched) else 0
    ctr = (clks / imps) if imps else 0.0
    convs = len(conv_log)
    cvr = (convs / clks) if clks else 0.0
    hot_imps = int(((event_log_enriched["campaign_id"] == hot_campaign_id) & (event_log_enriched["event_type"] == "impression")).sum())
    hot_share_real = (hot_imps / imps) if imps else 0.0

    distinct_users = int(event_log_enriched["user_id"].nunique())
    distinct_ads = int(event_log_enriched["ad_id"].nunique())
    distinct_campaigns = int(event_log_enriched["campaign_id"].nunique())

    print("=== ODS generation summary ===")
    print(f"Output dir: {out_dir}")
    print(f"Date range: {args.start_dt} .. {end_dt_str} (days={args.days})")
    print(f"Event rows: {len(event_log_enriched):,} (imps={imps:,}, clicks={clks:,}, CTR={ctr:.2%})")
    print(f"Conversion rows: {convs:,} (click->conv={cvr:.2%})")
    print(f"Cost rows: {len(cost_df):,}")
    print(f"Distinct: users={distinct_users:,}, ads={distinct_ads:,}, campaigns={distinct_campaigns:,}")
    print(f"Hot campaign: {hot_campaign_id} impressions share={hot_share_real:.2%}")
    print("")
    print("Files written:")
    for p, n in paths_written:
        print(f"- {p} ({n:,} rows)")

    # ===================== Product distribution (from ods_ad_meta.csv) =====================
    # share = occurrences / all non-empty product_id occurrences
    if "product_id" in ods_ad_meta.columns:
        pid = ods_ad_meta["product_id"].fillna("").astype(str)
        nonempty = pid.ne("")
        total_nonempty = int(nonempty.sum())
        distinct_products = int(pid[nonempty].nunique())
        vc = pid[nonempty].value_counts()

        top10_share = float(vc.iloc[:10].sum() / total_nonempty) if total_nonempty else 0.0
        top100_share = float(vc.iloc[:100].sum() / total_nonempty) if total_nonempty else 0.0

        print("")
        print(f"[ProductDist] distinct_products={distinct_products:,}")
        if total_nonempty:
            print("[ProductDist] top10_counts:")
            for k, v in vc.head(10).items():
                print(f"[ProductDist] - {k}: {int(v)}")
        print(f"[ProductDist] top10_share={top10_share:.2%}")
        print(f"[ProductDist] top100_share={top100_share:.2%}")


if __name__ == "__main__":
    main()

