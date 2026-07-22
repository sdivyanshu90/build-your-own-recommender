"""Cleaning, label construction, and leakage-safe temporal splits."""

from pathlib import Path

import pandas as pd

from recommender.artifacts.manifest import ArtifactManifest
from recommender.config.models import AppConfig
from recommender.data.schemas import EVENT_WEIGHTS
from recommender.data.validation import validate_frames
from recommender.exceptions import DataQualityError


def clean_events(users: pd.DataFrame, items: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic invalid-record and duplicate policies."""
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    valid = (
        frame["user_id"].notna()
        & frame["item_id"].notna()
        & frame["event_type"].isin(EVENT_WEIGHTS)
        & (frame["position"].fillna(0) >= 1)
        & frame["timestamp"].notna()
        & frame["user_id"].isin(users["user_id"])
        & frame["item_id"].isin(items["item_id"])
    )
    return (
        frame.loc[valid]
        .sort_values(["timestamp", "event_id"])
        .drop_duplicates("event_id", keep="last")
        .reset_index(drop=True)
    )


def temporal_split(
    events: pd.DataFrame, train_fraction: float, validation_fraction: float
) -> dict[str, pd.DataFrame]:
    """Split on global event-time boundaries; no event crosses backward in time."""
    if events.empty:
        raise DataQualityError("no valid events remain after cleaning")
    ordered = events.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    train_end = max(1, int(len(ordered) * train_fraction))
    validation_end = max(train_end + 1, int(len(ordered) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(ordered) - 1)
    return {
        "train": ordered.iloc[:train_end].copy(),
        "validation": ordered.iloc[train_end:validation_end].copy(),
        "test": ordered.iloc[validation_end:].copy(),
    }


def prepare_data(config: AppConfig, version: str = "dataset-v001") -> Path:
    raw = config.paths.data_dir / "raw"
    users = pd.read_parquet(raw / "users.parquet")
    items = pd.read_parquet(raw / "items.parquet")
    events = pd.read_parquet(raw / "events.parquet")
    findings = validate_frames(users, items, events)
    cleaned = clean_events(users, items, events)
    cleaned["event_weight"] = cleaned["event_type"].map(EVENT_WEIGHTS).astype(float)
    cleaned["label"] = (
        cleaned["event_type"].isin(config.data.positive_events)
        & (cleaned["event_weight"] >= config.data.positive_weight_threshold)
    ).astype("int8")
    splits = temporal_split(cleaned, config.data.train_fraction, config.data.validation_fraction)
    directory = config.paths.artifact_dir / "datasets" / version
    directory.mkdir(parents=True, exist_ok=True)
    users.to_parquet(directory / "users.parquet", index=False)
    items.to_parquet(directory / "items.parquet", index=False)
    for name, split in splits.items():
        split.to_parquet(directory / f"{name}.parquet", index=False)
    metadata = {
        "seed": config.seed,
        "row_counts": {name: len(frame) for name, frame in splits.items()},
        "positive_counts": {name: int(frame["label"].sum()) for name, frame in splits.items()},
        "split_boundaries": {
            name: {"min": str(frame["timestamp"].min()), "max": str(frame["timestamp"].max())}
            for name, frame in splits.items()
        },
        "quality_findings": [finding.__dict__ for finding in findings],
    }
    manifest = ArtifactManifest.create(
        "dataset",
        version,
        config.data.model_dump(mode="json"),
        list(cleaned.columns),
        metadata=metadata,
    )
    manifest.write(directory)
    return directory
