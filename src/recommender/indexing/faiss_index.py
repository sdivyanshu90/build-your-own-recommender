"""Optional FAISS inner-product index."""

from pathlib import Path
from typing import Any

import numpy as np

from recommender.exceptions import ArtifactError


class FaissIndex:
    def __init__(self, index: Any, item_ids: np.ndarray, dimension: int) -> None:
        self.index = index
        self.item_ids = np.asarray(item_ids, dtype=str)
        self.dimension = dimension

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        item_ids: np.ndarray,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> "FaissIndex":
        try:
            import faiss
        except ImportError as error:
            raise ArtifactError("FAISS backend requested; install the 'ann' extra") from error
        index = faiss.IndexHNSWFlat(embeddings.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        return cls(index, item_ids, embeddings.shape[1])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        limit = min(k, len(self.item_ids))
        scores, positions = self.index.search(
            np.ascontiguousarray(queries, dtype=np.float32), limit
        )
        return scores, self.item_ids[positions]

    def save(self, path: Path) -> None:
        import faiss

        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, item_ids: np.ndarray, dimension: int) -> "FaissIndex":
        import faiss

        index = faiss.read_index(str(path))
        if index.d != dimension or index.ntotal != len(item_ids):
            raise ArtifactError("FAISS index metadata does not match its manifest")
        return cls(index, item_ids, dimension)
