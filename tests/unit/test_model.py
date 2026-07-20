import pytest
import torch

from recommender.config.models import ModelConfig
from recommender.models.losses import InBatchSoftmaxLoss
from recommender.models.two_tower import TwoTowerModel


def _vocabularies():
    namespaces = [
        "user.user_id",
        "user.age_bucket",
        "user.country",
        "user.language",
        "user.subscription_tier",
        "user.device_preference",
        "user.preferred_categories",
        "context.device",
        "item.item_id",
        "item.category",
        "item.subcategory",
        "item.language",
        "item.brand",
        "item.price_bucket",
    ]
    return {name: 10 for name in namespaces}


def _batch(size=4):
    batch = {}
    for field in (
        "user_id",
        "age_bucket",
        "country",
        "language",
        "subscription_tier",
        "device_preference",
    ):
        batch[f"user_{field}_idx"] = torch.ones(size, dtype=torch.long)
    for field in ("item_id", "category", "subcategory", "language", "brand", "price_bucket"):
        batch[f"item_{field}_idx"] = torch.ones(size, dtype=torch.long)
    batch["user_preferences_idx"] = torch.ones((size, 2), dtype=torch.long)
    batch["context_device_idx"] = torch.ones(size, dtype=torch.long)
    batch["user_numerical"] = torch.zeros((size, 3))
    batch["item_numerical"] = torch.zeros((size, 3))
    return batch


def test_towers_return_normalized_embeddings() -> None:
    model = TwoTowerModel(
        _vocabularies(), ModelConfig(embedding_dim=8, hidden_dims=(12,), dropout=0)
    )
    users, items = model(_batch())
    assert users.shape == items.shape == (4, 8)
    torch.testing.assert_close(users.norm(dim=1), torch.ones(4))
    torch.testing.assert_close(items.norm(dim=1), torch.ones(4))


def test_loss_prefers_aligned_pairs() -> None:
    embeddings = torch.eye(4)
    item_ids = torch.arange(4)
    loss = InBatchSoftmaxLoss(0.1)
    aligned = loss(embeddings, embeddings, item_ids)
    misaligned = loss(embeddings, embeddings.flip(0), item_ids)
    assert aligned < misaligned


def test_duplicate_items_are_multi_positive() -> None:
    users = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    items = users.clone()
    value = InBatchSoftmaxLoss()(users, items, torch.tensor([5, 5]))
    assert torch.isfinite(value)
    assert value >= 0


def test_smoothed_symmetric_loss_and_invalid_temperature() -> None:
    embeddings = torch.eye(3)
    value = InBatchSoftmaxLoss(0.2, label_smoothing=0.1, symmetric=True)(
        embeddings, embeddings, torch.arange(3)
    )
    assert torch.isfinite(value)
    with pytest.raises(ValueError, match="temperature"):
        InBatchSoftmaxLoss(0)
