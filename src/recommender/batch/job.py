"""Restartable bounded-memory Parquet batch recommendation job."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from recommender.artifacts.manifest import ArtifactManifest
from recommender.config.models import AppConfig
from recommender.retrieval.runtime import RecommendationRuntime
from recommender.utils.io import atomic_write_json


def batch_recommend(
    config: AppConfig,
    input_path: Path,
    output_version: str = "batch-v001",
    top_k: int | None = None,
) -> Path:
    runtime = RecommendationRuntime.load(config)
    output_dir = config.paths.artifact_dir / "batch-recommendations" / output_version
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(input_path)
    total_users = 0
    fallback_users = 0
    for index, record_batch in enumerate(parquet.iter_batches(batch_size=512)):
        part_path = output_dir / f"part-{index:05d}.parquet"
        if part_path.exists():
            continue
        frame = record_batch.to_pandas()
        if "user_id" not in frame:
            raise ValueError("batch input requires a user_id column")
        rows = []
        for user_id in frame["user_id"].astype(str):
            result = runtime.recommend(user_id, top_k or config.serving.default_k)
            fallback_users += int(result.fallback_reason is not None)
            rows.append(
                {
                    "user_id": user_id,
                    "item_ids": [candidate.item_id for candidate in result.candidates],
                    "scores": [candidate.retrieval_score for candidate in result.candidates],
                    "fallback_reason": result.fallback_reason,
                    "model_version": runtime.model_version,
                    "index_version": runtime.index_version,
                }
            )
        pd.DataFrame(rows).to_parquet(part_path, index=False)
        total_users += len(rows)
        atomic_write_json(
            output_dir / "checkpoint.json",
            {"last_completed_part": index, "users_processed_this_run": total_users},
        )
    manifest = ArtifactManifest.create(
        "batch",
        output_version,
        {"top_k": top_k or config.serving.default_k},
        ["user_id", "item_ids", "scores", "fallback_reason", "model_version", "index_version"],
        dependencies={"model": runtime.model_version, "index": runtime.index_version},
        metadata={
            "users_processed_this_run": total_users,
            "fallback_users_this_run": fallback_users,
        },
    )
    manifest.write(output_dir)
    return output_dir
