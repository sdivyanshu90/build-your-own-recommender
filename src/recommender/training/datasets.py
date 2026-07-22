"""Tensor-ready datasets and model input conversion."""

from collections.abc import Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset

USER_FIELDS = (
    "user_id",
    "age_bucket",
    "country",
    "language",
    "subscription_tier",
    "device_preference",
)
ITEM_FIELDS = ("item_id", "category", "subcategory", "language", "brand", "price_bucket")


def user_frame_to_batch(frame: pd.DataFrame) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        f"user_{field}_idx": torch.as_tensor(
            frame[f"user_{field}_idx"].to_numpy(), dtype=torch.long
        )
        for field in USER_FIELDS
    }
    preference_columns = sorted(
        column for column in map(str, frame.columns) if column.startswith("user_preference_")
    )
    batch["user_preferences_idx"] = torch.as_tensor(
        frame[preference_columns].to_numpy(), dtype=torch.long
    )
    batch["context_device_idx"] = torch.as_tensor(
        frame["context_device_idx"].to_numpy(), dtype=torch.long
    )
    batch["user_numerical"] = torch.as_tensor(
        frame[
            ["user_account_age_days_z", "user_activity_score_z", "context_position_z"]
        ].to_numpy(),
        dtype=torch.float32,
    )
    return batch


def item_frame_to_batch(frame: pd.DataFrame) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        f"item_{field}_idx": torch.as_tensor(
            frame[f"item_{field}_idx"].to_numpy(), dtype=torch.long
        )
        for field in ITEM_FIELDS
    }
    batch["item_numerical"] = torch.as_tensor(
        frame[["item_price_z", "item_popularity_z", "item_freshness_days_z"]].to_numpy(),
        dtype=torch.float32,
    )
    batch["item_identity"] = batch["item_item_id_idx"]
    return batch


def frame_to_batch(frame: pd.DataFrame) -> dict[str, torch.Tensor]:
    return {**user_frame_to_batch(frame), **item_frame_to_batch(frame)}


class InteractionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, positives_only: bool = True) -> None:
        selected = frame.loc[frame["label"].eq(1)] if positives_only else frame
        if selected.empty:
            raise ValueError("interaction dataset contains no positive examples")
        self.batch = frame_to_batch(selected.reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.batch["item_identity"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.batch.items()}


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
