"""Batched item-tower embedding export."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recommender.artifacts.manifest import ArtifactManifest
from recommender.config.models import AppConfig
from recommender.features.processor import FeatureProcessor
from recommender.training.datasets import item_frame_to_batch, move_batch
from recommender.training.trainer import load_model


def export_item_embeddings(
    config: AppConfig,
    dataset_version: str = "dataset-v001",
    feature_version: str = "features-v001",
    model_version: str = "model-v001",
    embedding_version: str = "embeddings-v001",
) -> Path:
    feature_dir = config.paths.artifact_dir / "feature-pipelines" / feature_version
    model_dir = config.paths.artifact_dir / "models" / model_version
    processor = FeatureProcessor.load(feature_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_dir, processor, config, device)
    items = pd.read_parquet(
        config.paths.artifact_dir / "datasets" / dataset_version / "transformed" / "items.parquet"
    )
    eligible = items.loc[items["available"]].reset_index(drop=True)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(eligible), config.training.batch_size):
            frame = eligible.iloc[start : start + config.training.batch_size]
            batch = move_batch(item_frame_to_batch(frame), device)
            batches.append(model.item_tower(batch).cpu().numpy().astype(np.float32))
    embeddings = (
        np.concatenate(batches)
        if batches
        else np.empty((0, config.model.embedding_dim), np.float32)
    )
    directory = config.paths.artifact_dir / "embeddings" / embedding_version
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "item_embeddings.npy", embeddings, allow_pickle=False)
    eligible[
        ["item_id_raw", "category_raw", "popularity_raw", "freshness_days_raw", "available"]
    ].to_parquet(directory / "items.parquet", index=False)
    manifest = ArtifactManifest.create(
        "embeddings",
        embedding_version,
        {"batch_size": config.training.batch_size},
        {"dimension": config.model.embedding_dim, "dtype": "float32"},
        dependencies={
            "dataset": dataset_version,
            "features": feature_version,
            "model": model_version,
        },
        metadata={
            "item_count": len(eligible),
            "embedding_dimension": config.model.embedding_dim,
            "similarity": config.model.similarity,
            "mean_norm": float(np.linalg.norm(embeddings, axis=1).mean())
            if len(embeddings)
            else 0.0,
        },
    )
    manifest.write(directory)
    return directory
