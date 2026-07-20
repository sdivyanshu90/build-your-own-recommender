"""Index construction, validation, persistence, and safe loading."""

from pathlib import Path

import numpy as np
import pandas as pd

from recommender.artifacts.manifest import ArtifactManifest
from recommender.artifacts.publication import publish_current
from recommender.config.models import AppConfig
from recommender.exceptions import ArtifactError, CompatibilityError
from recommender.indexing.base import VectorIndex
from recommender.indexing.exact import ExactIndex
from recommender.indexing.faiss_index import FaissIndex


def build_index(
    config: AppConfig,
    embedding_version: str = "embeddings-v001",
    index_version: str = "index-v001",
) -> Path:
    embedding_dir = config.paths.artifact_dir / "embeddings" / embedding_version
    embedding_manifest = ArtifactManifest.load(embedding_dir)
    embeddings = np.load(embedding_dir / "item_embeddings.npy", allow_pickle=False)
    items = pd.read_parquet(embedding_dir / "items.parquet")
    item_ids = np.asarray(items["item_id_raw"].astype(str).tolist(), dtype=str)
    if embeddings.shape != (len(item_ids), config.model.embedding_dim):
        raise CompatibilityError(
            "embedding matrix shape does not match model dimension or item metadata"
        )
    directory = config.paths.artifact_dir / "indexes" / index_version
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "item_ids.npy", item_ids, allow_pickle=False)
    if config.index.backend == "faiss":
        index: VectorIndex = FaissIndex.build(
            embeddings,
            item_ids,
            config.index.hnsw_m,
            config.index.hnsw_ef_construction,
            config.index.hnsw_ef_search,
        )
        index.save(directory / "index.faiss")  # type: ignore[attr-defined]
    else:
        np.save(directory / "vectors.npy", embeddings, allow_pickle=False)
        index = ExactIndex(embeddings, item_ids)
    sample_count = min(100, len(embeddings))
    queries = embeddings[:sample_count]
    _, found = index.search(queries, 1)
    self_recall = float(np.mean(found[:, 0] == item_ids[:sample_count])) if sample_count else 1.0
    if self_recall < 0.99:
        raise ArtifactError(f"index validation failed: self recall={self_recall:.4f}")
    manifest = ArtifactManifest.create(
        "index",
        index_version,
        config.index.model_dump(mode="json"),
        {"dimension": config.model.embedding_dim, "metric": config.index.metric},
        dependencies={
            "embeddings": embedding_version,
            "model": embedding_manifest.dependencies["model"],
        },
        metadata={
            "backend": config.index.backend,
            "item_count": len(item_ids),
            "dimension": config.model.embedding_dim,
            "metric": config.index.metric,
            "self_recall_at_1": self_recall,
        },
    )
    manifest.write(directory)
    publish_current(directory, config.paths.artifact_dir / "indexes")
    return directory


def load_index(directory: Path, expected_model: str | None = None) -> VectorIndex:
    manifest = ArtifactManifest.load(directory)
    if manifest.artifact_type != "index":
        raise ArtifactError("artifact is not an index")
    if expected_model is not None:
        manifest.require_dependency("model", expected_model)
    item_ids = np.load(directory / "item_ids.npy", allow_pickle=False)
    dimension = int(manifest.metadata["dimension"])
    if manifest.metadata["backend"] == "faiss":
        return FaissIndex.load(directory / "index.faiss", item_ids, dimension)
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    return ExactIndex(vectors, item_ids)
