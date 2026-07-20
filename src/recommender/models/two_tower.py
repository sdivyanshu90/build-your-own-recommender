"""Independent user and item towers with normalized retrieval embeddings."""

from collections.abc import Mapping, Sequence
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from recommender.config.models import ModelConfig


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[name]


class Projection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        dimensions = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(pairwise(dimensions)):
            layers.append(nn.Linear(source, target))
            if index < len(dimensions) - 2:
                layers.extend([nn.LayerNorm(target), _activation(activation), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.network(inputs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("projection must return a tensor")
        return output


class UserTower(nn.Module):
    categorical_fields = (
        "user_id",
        "age_bucket",
        "country",
        "language",
        "subscription_tier",
        "device_preference",
    )

    def __init__(self, vocabulary_sizes: Mapping[str, int], config: ModelConfig) -> None:
        super().__init__()
        self.similarity = config.similarity
        self.embeddings = nn.ModuleDict()
        dimensions = 0
        for field in self.categorical_fields:
            dimension = (
                config.id_embedding_dim if field == "user_id" else config.categorical_embedding_dim
            )
            self.embeddings[field] = nn.Embedding(
                vocabulary_sizes[f"user.{field}"], dimension, padding_idx=0
            )
            dimensions += dimension
        self.preference_embedding = nn.Embedding(
            vocabulary_sizes["user.preferred_categories"],
            config.categorical_embedding_dim,
            padding_idx=0,
        )
        self.context_embedding = nn.Embedding(
            vocabulary_sizes["context.device"], config.categorical_embedding_dim, padding_idx=0
        )
        dimensions += config.categorical_embedding_dim * 2 + 3
        self.projection = Projection(
            dimensions, config.hidden_dims, config.embedding_dim, config.dropout, config.activation
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.embeddings[field](batch[f"user_{field}_idx"]) for field in self.categorical_fields
        ]
        preferences = batch["user_preferences_idx"]
        mask = preferences.ne(0).unsqueeze(-1)
        preference_sum = (self.preference_embedding(preferences) * mask).sum(dim=1)
        preference_count = mask.sum(dim=1).clamp_min(1)
        parts.append(preference_sum / preference_count)
        parts.append(self.context_embedding(batch["context_device_idx"]))
        parts.append(batch["user_numerical"])
        embedding = self.projection(torch.cat(parts, dim=-1))
        return F.normalize(embedding, dim=-1) if self.similarity == "cosine" else embedding


class ItemTower(nn.Module):
    categorical_fields = ("item_id", "category", "subcategory", "language", "brand", "price_bucket")

    def __init__(self, vocabulary_sizes: Mapping[str, int], config: ModelConfig) -> None:
        super().__init__()
        self.similarity = config.similarity
        self.embeddings = nn.ModuleDict()
        dimensions = 3
        for field in self.categorical_fields:
            dimension = (
                config.id_embedding_dim if field == "item_id" else config.categorical_embedding_dim
            )
            self.embeddings[field] = nn.Embedding(
                vocabulary_sizes[f"item.{field}"], dimension, padding_idx=0
            )
            dimensions += dimension
        self.projection = Projection(
            dimensions, config.hidden_dims, config.embedding_dim, config.dropout, config.activation
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.embeddings[field](batch[f"item_{field}_idx"]) for field in self.categorical_fields
        ]
        parts.append(batch["item_numerical"])
        embedding = self.projection(torch.cat(parts, dim=-1))
        return F.normalize(embedding, dim=-1) if self.similarity == "cosine" else embedding


class TwoTowerModel(nn.Module):
    def __init__(self, vocabulary_sizes: Mapping[str, int], config: ModelConfig) -> None:
        super().__init__()
        self.user_tower = UserTower(vocabulary_sizes, config)
        self.item_tower = ItemTower(vocabulary_sizes, config)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(batch), self.item_tower(batch)

    @staticmethod
    def similarity(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        return user_embeddings @ item_embeddings.T
