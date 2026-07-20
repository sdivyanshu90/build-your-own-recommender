"""Deterministic synthetic data with preference and exposure structure."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.config.models import AppConfig
from recommender.data.schemas import EVENT_COLUMNS, ITEM_COLUMNS, USER_COLUMNS

CATEGORIES = np.array(["books", "electronics", "fashion", "fitness", "food", "music"])
COUNTRIES = np.array(["IN", "US", "GB", "DE", "BR"])
LANGUAGES = np.array(["en", "hi", "de", "pt"])
DEVICES = np.array(["mobile", "desktop", "tablet", "tv"])


def _users(config: AppConfig, rng: np.random.Generator) -> pd.DataFrame:
    count = config.data.num_users
    clusters = rng.integers(0, len(CATEGORIES), count)
    activity = np.clip(rng.lognormal(0, 1, count), 0.1, 20)
    records = []
    for index in range(count):
        primary = str(CATEGORIES[clusters[index]])
        secondary = str(CATEGORIES[(clusters[index] + 1) % len(CATEGORIES)])
        records.append(
            {
                "user_id": f"u{index:06d}",
                "age_bucket": str(rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"])),
                "country": str(rng.choice(COUNTRIES, p=[0.35, 0.25, 0.15, 0.1, 0.15])),
                "language": str(rng.choice(LANGUAGES, p=[0.55, 0.2, 0.1, 0.15])),
                "subscription_tier": str(
                    rng.choice(["free", "standard", "premium"], p=[0.6, 0.3, 0.1])
                ),
                "device_preference": str(rng.choice(DEVICES, p=[0.58, 0.27, 0.08, 0.07])),
                "account_age_days": int(rng.integers(1, 2500)),
                "preferred_categories": f"{primary}|{secondary}",
                "activity_score": float(activity[index]),
                "_cluster": int(clusters[index]),
            }
        )
    users = pd.DataFrame(records)
    missing = rng.choice(count, max(1, count // 50), replace=False)
    users.loc[missing, "language"] = None
    return users


def _items(config: AppConfig, rng: np.random.Generator, end: datetime) -> pd.DataFrame:
    count = config.data.num_items
    ranks = np.arange(1, count + 1)
    popularity = 1 / np.power(ranks, 1.15)
    popularity /= popularity.sum()
    rng.shuffle(popularity)
    records = []
    for index in range(count):
        category = str(CATEGORIES[index % len(CATEGORIES)])
        age = int(rng.integers(0, 730))
        price = float(np.round(rng.lognormal(3.0, 0.8), 2))
        records.append(
            {
                "item_id": f"i{index:06d}",
                "category": category,
                "subcategory": f"{category}-{index % 4}",
                "language": str(rng.choice(LANGUAGES, p=[0.6, 0.17, 0.1, 0.13])),
                "brand": f"brand-{index % 25}",
                "price_bucket": "low" if price < 15 else "medium" if price < 50 else "high",
                "price": price,
                "popularity": float(popularity[index]),
                "freshness_days": age,
                "title": f"{category.title()} item {index}",
                "available": bool(rng.random() > 0.04),
                "created_at": end - timedelta(days=age),
                "_category_index": index % len(CATEGORIES),
            }
        )
    items = pd.DataFrame(records)
    missing = rng.choice(count, max(1, count // 70), replace=False)
    items.loc[missing, "brand"] = None
    return items


def generate_synthetic(config: AppConfig, output_dir: Path | None = None) -> dict[str, Path]:
    """Generate reproducible users, items, and biased event history as Parquet."""
    rng = np.random.default_rng(config.seed)
    end = datetime(2025, 1, 1, tzinfo=UTC)
    start = end - timedelta(days=120)
    users = _users(config, rng)
    items = _items(config, rng, end)
    user_weights = users["activity_score"].to_numpy(dtype=float)
    user_weights /= user_weights.sum()
    base_item_weights = items["popularity"].to_numpy(dtype=float)
    cold_user_start = int(len(users) * 0.9)
    cold_item_start = int(len(items) * 0.9)
    records: list[dict[str, object]] = []
    for event_index in range(config.data.num_events):
        fraction = event_index / config.data.num_events
        user_pool = len(users) if fraction > 0.85 else cold_user_start
        allowed_user_weights = user_weights[:user_pool]
        allowed_user_weights /= allowed_user_weights.sum()
        user_index = int(rng.choice(user_pool, p=allowed_user_weights))
        item_pool = len(items) if fraction > 0.85 else cold_item_start
        affinity = np.ones(item_pool)
        affinity[
            items["_category_index"].to_numpy()[:item_pool] == users.iloc[user_index]["_cluster"]
        ] = 8
        weights = base_item_weights[:item_pool] * affinity
        weights /= weights.sum()
        item_index = int(rng.choice(item_pool, p=weights))
        position = int(np.clip(rng.geometric(0.25), 1, 50))
        match = int(users.iloc[user_index]["_cluster"] == items.iloc[item_index]["_category_index"])
        positive_probability = np.clip(0.08 + 0.48 * match - 0.008 * position, 0.01, 0.75)
        if rng.random() < positive_probability:
            event_type = str(
                rng.choice(
                    ["click", "view", "add_to_cart", "purchase", "rating"],
                    p=[0.38, 0.33, 0.13, 0.1, 0.06],
                )
            )
        else:
            event_type = "impression"
        timestamp = start + timedelta(seconds=fraction * 120 * 86400 + float(rng.uniform(0, 600)))
        records.append(
            {
                "event_id": f"e{event_index:09d}",
                "user_id": users.iloc[user_index]["user_id"],
                "item_id": items.iloc[item_index]["item_id"],
                "event_type": event_type,
                "timestamp": timestamp,
                "session_id": f"s{user_index:06d}-{int(timestamp.timestamp()) // 1800}",
                "position": position,
                "device": users.iloc[user_index]["device_preference"]
                if rng.random() > 0.15
                else str(rng.choice(DEVICES)),
                "dwell_seconds": float(rng.gamma(2 + 3 * match, 8))
                if event_type != "impression"
                else 0.0,
                "rating": float(rng.integers(3, 6)) if event_type == "rating" else np.nan,
            }
        )
    events = pd.DataFrame(records)
    duplicate_count = int(len(events) * config.data.duplicate_fraction)
    if duplicate_count:
        events = pd.concat(
            [events, events.sample(duplicate_count, random_state=config.seed)], ignore_index=True
        )
    invalid_count = int(config.data.num_events * config.data.invalid_fraction)
    if invalid_count:
        invalid = events.iloc[:invalid_count].copy()
        invalid["event_id"] = [f"invalid-{index}" for index in range(invalid_count)]
        invalid.loc[invalid.index[::3], "user_id"] = None
        invalid.loc[invalid.index[1::3], "event_type"] = "unsupported"
        invalid.loc[invalid.index[2::3], "position"] = -1
        events = pd.concat([events, invalid], ignore_index=True)
    directory = output_dir or config.paths.data_dir / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{name}.parquet" for name in ("users", "items", "events")}
    users.loc[:, USER_COLUMNS].to_parquet(paths["users"], index=False)
    items.loc[:, ITEM_COLUMNS].to_parquet(paths["items"], index=False)
    events.loc[:, EVENT_COLUMNS].to_parquet(paths["events"], index=False)
    return paths
