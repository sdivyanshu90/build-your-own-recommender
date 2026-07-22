"""Versioned train-fitted feature transformations shared by training and serving."""

from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from recommender.artifacts.manifest import ArtifactManifest
from recommender.config.models import AppConfig
from recommender.exceptions import ArtifactError
from recommender.utils.io import atomic_write_json

PAD_TOKEN = "<PAD>"  # noqa: S105  # nosec B105
UNKNOWN_TOKEN = "<UNK>"  # noqa: S105  # nosec B105


class NumericalStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean: float
    std: float = Field(gt=0)


class FeatureProcessor(BaseModel):
    """Serializable vocabularies and numerical statistics fitted on training rows only."""

    model_config = ConfigDict(extra="forbid")

    VERSION: ClassVar[str] = "1"
    version: str
    min_frequency: int
    max_history: int
    vocabularies: dict[str, dict[str, int]]
    numerical: dict[str, NumericalStats]

    user_categorical: ClassVar[tuple[str, ...]] = (
        "user_id",
        "age_bucket",
        "country",
        "language",
        "subscription_tier",
        "device_preference",
    )
    item_categorical: ClassVar[tuple[str, ...]] = (
        "item_id",
        "category",
        "subcategory",
        "language",
        "brand",
        "price_bucket",
    )
    user_numerical: ClassVar[tuple[str, ...]] = ("account_age_days", "activity_score")
    item_numerical: ClassVar[tuple[str, ...]] = ("price", "popularity", "freshness_days")

    @staticmethod
    def _vocabulary(values: list[str], min_frequency: int) -> dict[str, int]:
        counts = Counter(
            value
            for value in values
            if value and value != "nan" and value not in {PAD_TOKEN, UNKNOWN_TOKEN}
        )
        tokens = sorted(token for token, count in counts.items() if count >= min_frequency)
        return {
            PAD_TOKEN: 0,
            UNKNOWN_TOKEN: 1,
            **{token: index + 2 for index, token in enumerate(tokens)},
        }

    @classmethod
    def fit(
        cls,
        users: pd.DataFrame,
        items: pd.DataFrame,
        train_events: pd.DataFrame,
        config: AppConfig,
        version: str = "features-v001",
    ) -> "FeatureProcessor":
        train_users = users[users["user_id"].isin(train_events["user_id"].unique())]
        train_items = items[items["item_id"].isin(train_events["item_id"].unique())]
        vocabularies: dict[str, dict[str, int]] = {}
        for field in cls.user_categorical:
            vocabularies[f"user.{field}"] = cls._vocabulary(
                train_users[field].fillna(UNKNOWN_TOKEN).astype(str).tolist(),
                config.features.min_frequency,
            )
        for field in cls.item_categorical:
            vocabularies[f"item.{field}"] = cls._vocabulary(
                train_items[field].fillna(UNKNOWN_TOKEN).astype(str).tolist(),
                config.features.min_frequency,
            )
        preferences = [
            token
            for value in train_users["preferred_categories"].fillna("").astype(str)
            for token in value.split("|")
            if token
        ]
        vocabularies["user.preferred_categories"] = cls._vocabulary(
            preferences, config.features.min_frequency
        )
        vocabularies["context.device"] = cls._vocabulary(
            train_events["device"].fillna(UNKNOWN_TOKEN).astype(str).tolist(),
            config.features.min_frequency,
        )
        numerical: dict[str, NumericalStats] = {}
        for prefix, frame, fields in (
            ("user", train_users, cls.user_numerical),
            ("item", train_items, cls.item_numerical),
        ):
            for field in fields:
                values = pd.to_numeric(frame[field], errors="coerce")
                mean = float(values.mean()) if values.notna().any() else 0.0
                std = float(values.std(ddof=0)) if values.notna().any() else 1.0
                numerical[f"{prefix}.{field}"] = NumericalStats(mean=mean, std=max(std, 1e-8))
        return cls(
            version=version,
            min_frequency=config.features.min_frequency,
            max_history=config.features.max_history,
            vocabularies=vocabularies,
            numerical=numerical,
        )

    def encode(self, namespace: str, value: Any) -> int:
        vocabulary = self.vocabularies[namespace]
        token = UNKNOWN_TOKEN if value is None or pd.isna(value) else str(value)
        return vocabulary.get(token, vocabulary[UNKNOWN_TOKEN])

    def normalize(self, namespace: str, value: Any) -> float:
        stats = self.numerical[namespace]
        numeric = stats.mean if value is None or pd.isna(value) else float(value)
        return (numeric - stats.mean) / stats.std

    def transform_users(self, users: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=users.index)
        result["user_id_raw"] = users["user_id"].astype(str)
        for field in self.user_categorical:
            result[f"user_{field}_idx"] = users[field].map(partial(self.encode, f"user.{field}"))
        for field in self.user_numerical:
            result[f"user_{field}_z"] = users[field].map(partial(self.normalize, f"user.{field}"))
        preference_vocab = self.vocabularies["user.preferred_categories"]
        width = min(4, self.max_history)
        encoded = []
        for value in users["preferred_categories"].fillna("").astype(str):
            tokens = [preference_vocab.get(token, 1) for token in value.split("|") if token][:width]
            encoded.append(tokens + [0] * (width - len(tokens)))
        for index in range(width):
            result[f"user_preference_{index}_idx"] = [row[index] for row in encoded]
        return result.reset_index(drop=True)

    def transform_items(self, items: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=items.index)
        result["item_id_raw"] = items["item_id"].astype(str)
        for field in self.item_categorical:
            result[f"item_{field}_idx"] = items[field].map(partial(self.encode, f"item.{field}"))
        for field in self.item_numerical:
            result[f"item_{field}_z"] = items[field].map(partial(self.normalize, f"item.{field}"))
        result["available"] = items["available"].fillna(False).astype(bool)
        result["category_raw"] = items["category"].fillna(UNKNOWN_TOKEN).astype(str)
        result["popularity_raw"] = items["popularity"].fillna(0).astype(float)
        result["freshness_days_raw"] = items["freshness_days"].fillna(np.inf).astype(float)
        return result.reset_index(drop=True)

    def transform_interactions(
        self, users: pd.DataFrame, items: pd.DataFrame, events: pd.DataFrame
    ) -> pd.DataFrame:
        transformed_users = self.transform_users(users)
        transformed_items = self.transform_items(items)
        result = events.merge(
            transformed_users, left_on="user_id", right_on="user_id_raw", how="inner"
        )
        result = result.merge(
            transformed_items, left_on="item_id", right_on="item_id_raw", how="inner"
        )
        result["context_device_idx"] = result["device"].map(
            lambda value: self.encode("context.device", value)
        )
        result["context_position_z"] = np.log1p(result["position"].astype(float)) / np.log(51)
        return result

    def save(self, directory: Path, config: AppConfig, dataset_version: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "processor.json", self.model_dump(mode="json"))
        manifest = ArtifactManifest.create(
            "features",
            self.version,
            config.features.model_dump(mode="json"),
            {key: len(value) for key, value in self.vocabularies.items()},
            dependencies={"dataset": dataset_version},
            metadata={"processor_contract": self.VERSION},
        )
        manifest.write(directory)

    @classmethod
    def load(cls, directory: Path) -> "FeatureProcessor":
        ArtifactManifest.load(directory)
        try:
            return cls.model_validate_json(
                (directory / "processor.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ArtifactError(f"invalid feature processor: {directory}") from error

    def vocabulary_sizes(self) -> dict[str, int]:
        return {key: len(value) for key, value in self.vocabularies.items()}


def fit_and_transform_features(
    config: AppConfig,
    dataset_version: str = "dataset-v001",
    feature_version: str = "features-v001",
) -> tuple[Path, Path]:
    dataset_dir = config.paths.artifact_dir / "datasets" / dataset_version
    dataset_manifest = ArtifactManifest.load(dataset_dir)
    users = pd.read_parquet(dataset_dir / "users.parquet")
    items = pd.read_parquet(dataset_dir / "items.parquet")
    train = pd.read_parquet(dataset_dir / "train.parquet")
    processor = FeatureProcessor.fit(users, items, train, config, feature_version)
    feature_dir = config.paths.artifact_dir / "feature-pipelines" / feature_version
    processor.save(feature_dir, config, dataset_manifest.version)
    transformed_dir = dataset_dir / "transformed"
    transformed_dir.mkdir(exist_ok=True)
    processor.transform_users(users).to_parquet(transformed_dir / "users.parquet", index=False)
    processor.transform_items(items).to_parquet(transformed_dir / "items.parquet", index=False)
    for split in ("train", "validation", "test"):
        events = pd.read_parquet(dataset_dir / f"{split}.parquet")
        processor.transform_interactions(users, items, events).to_parquet(
            transformed_dir / f"{split}.parquet", index=False
        )
    return feature_dir, transformed_dir
