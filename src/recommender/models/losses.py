"""Retrieval objectives with duplicate-positive masking."""

import torch
from torch import nn
from torch.nn import functional as F


class InBatchSoftmaxLoss(nn.Module):
    """Multi-positive InfoNCE loss; repeated item IDs are positives, not false negatives."""

    def __init__(
        self, temperature: float = 0.07, label_smoothing: float = 0.0, symmetric: bool = False
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.symmetric = symmetric

    def _direction(self, logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        positive_loss = -(log_probabilities * positive_mask).sum(dim=1) / positive_count
        if self.label_smoothing:
            uniform_loss = -log_probabilities.mean(dim=1)
            positive_loss = (
                1 - self.label_smoothing
            ) * positive_loss + self.label_smoothing * uniform_loss
        return positive_loss.mean()

    def forward(
        self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        logits = user_embeddings @ item_embeddings.T / self.temperature
        mask = item_ids[:, None].eq(item_ids[None, :]).to(logits.dtype)
        loss = self._direction(logits, mask)
        if self.symmetric:
            loss = (loss + self._direction(logits.T, mask.T)) / 2
        return loss
