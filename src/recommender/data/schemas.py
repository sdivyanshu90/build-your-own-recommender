"""Canonical entity and event schemas."""

from typing import Final

USER_COLUMNS: Final = (
    "user_id",
    "age_bucket",
    "country",
    "language",
    "subscription_tier",
    "device_preference",
    "account_age_days",
    "preferred_categories",
    "activity_score",
)
ITEM_COLUMNS: Final = (
    "item_id",
    "category",
    "subcategory",
    "language",
    "brand",
    "price_bucket",
    "price",
    "popularity",
    "freshness_days",
    "title",
    "available",
    "created_at",
)
EVENT_COLUMNS: Final = (
    "event_id",
    "user_id",
    "item_id",
    "event_type",
    "timestamp",
    "session_id",
    "position",
    "device",
    "dwell_seconds",
    "rating",
)

EVENT_WEIGHTS: Final = {
    "impression": 0.0,
    "click": 1.0,
    "view": 1.2,
    "add_to_cart": 2.5,
    "purchase": 4.0,
    "rating": 1.5,
}
