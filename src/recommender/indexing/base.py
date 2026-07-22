"""Vector index protocol."""

from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    dimension: int
    item_ids: np.ndarray

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return descending scores and raw string item IDs."""
        ...
