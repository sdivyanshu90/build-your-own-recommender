"""Deterministic pluggable explicit negative samplers."""

from collections.abc import Callable, Sequence

import numpy as np

from recommender.exceptions import ConfigurationError


class UniformSampler:
    def __init__(self, item_ids: Sequence[int], seed: int) -> None:
        if not item_ids:
            raise ConfigurationError("negative sampler requires at least one item")
        self.items = np.asarray(item_ids, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def sample(
        self, positive: int, count: int, known_positives: set[int] | None = None
    ) -> np.ndarray:
        excluded = (known_positives or set()) | {positive}
        eligible = self.items[~np.isin(self.items, list(excluded))]
        if not len(eligible):
            raise ConfigurationError("no eligible negatives remain")
        return self.rng.choice(eligible, size=count, replace=len(eligible) < count)


class PopularitySampler(UniformSampler):
    def __init__(
        self, item_ids: Sequence[int], popularity: Sequence[float], seed: int, alpha: float = 0.75
    ) -> None:
        super().__init__(item_ids, seed)
        probabilities = np.asarray(popularity, dtype=float)
        if (
            len(probabilities) != len(self.items)
            or np.any(probabilities < 0)
            or probabilities.sum() <= 0
        ):
            raise ConfigurationError(
                "popularity must be non-negative, non-empty, and aligned with item_ids"
            )
        self.probabilities = np.power(probabilities, alpha)
        self.probabilities /= self.probabilities.sum()

    def sample(
        self, positive: int, count: int, known_positives: set[int] | None = None
    ) -> np.ndarray:
        excluded = (known_positives or set()) | {positive}
        mask = ~np.isin(self.items, list(excluded))
        eligible = self.items[mask]
        probabilities = self.probabilities[mask]
        if not len(eligible):
            raise ConfigurationError("no eligible negatives remain")
        probabilities /= probabilities.sum()
        return self.rng.choice(eligible, size=count, replace=len(eligible) < count, p=probabilities)


class HardNegativeSampler:
    """Select top ANN candidates while excluding positives and accidental positives."""

    def __init__(self, search: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]) -> None:
        self.search = search

    def sample(
        self,
        query_embedding: np.ndarray,
        positive: int,
        count: int,
        known_positives: set[int] | None = None,
    ) -> np.ndarray:
        excluded = (known_positives or set()) | {positive}
        _, candidates = self.search(query_embedding[None, :], count + len(excluded) + 20)
        selected = [int(candidate) for candidate in candidates[0] if int(candidate) not in excluded]
        if len(selected) < count:
            raise ConfigurationError("hard-negative search returned too few eligible candidates")
        return np.asarray(selected[:count], dtype=np.int64)
