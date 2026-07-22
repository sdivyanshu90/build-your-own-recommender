"""Brute-force correctness index."""

import numpy as np


class ExactIndex:
    def __init__(self, embeddings: np.ndarray, item_ids: np.ndarray) -> None:
        if embeddings.ndim != 2 or len(embeddings) != len(item_ids):
            raise ValueError("embeddings must be 2-D and aligned with item IDs")
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.item_ids = np.asarray(item_ids, dtype=str)
        self.dimension = self.embeddings.shape[1]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dimension:
            raise ValueError(f"query dimension must be {self.dimension}")
        limit = min(max(k, 0), len(self.item_ids))
        if limit == 0:
            return np.empty((len(queries), 0), np.float32), np.empty((len(queries), 0), dtype=str)
        similarities = queries @ self.embeddings.T
        positions = np.argpartition(-similarities, limit - 1, axis=1)[:, :limit]
        candidate_scores = np.take_along_axis(similarities, positions, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        positions = np.take_along_axis(positions, order, axis=1)
        scores = np.take_along_axis(similarities, positions, axis=1)
        return scores, self.item_ids[positions]
