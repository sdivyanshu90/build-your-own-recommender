"""Leakage-aware exact offline evaluation and baseline comparison."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recommender.artifacts.manifest import ArtifactManifest
from recommender.config.models import AppConfig
from recommender.evaluation.metrics import (
    catalog_coverage,
    category_diversity,
    novelty,
    ranking_metrics,
)
from recommender.features.processor import FeatureProcessor
from recommender.indexing.exact import ExactIndex
from recommender.training.datasets import move_batch, user_frame_to_batch
from recommender.training.trainer import load_model
from recommender.utils.io import atomic_write_json


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}


def evaluate_model(
    config: AppConfig,
    dataset_version: str = "dataset-v001",
    feature_version: str = "features-v001",
    model_version: str = "model-v001",
    embedding_version: str = "embeddings-v001",
) -> Path:
    root = config.paths.artifact_dir
    processor = FeatureProcessor.load(root / "feature-pipelines" / feature_version)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(root / "models" / model_version, processor, config, device)
    transformed = root / "datasets" / dataset_version / "transformed"
    train = pd.read_parquet(transformed / "train.parquet")
    validation = pd.read_parquet(transformed / "validation.parquet")
    test = pd.read_parquet(transformed / "test.parquet")
    embeddings_dir = root / "embeddings" / embedding_version
    embedding_manifest = ArtifactManifest.load(embeddings_dir)
    embedding_manifest.require_dependency("model", model_version)
    vectors = np.load(embeddings_dir / "item_embeddings.npy", allow_pickle=False)
    item_metadata = pd.read_parquet(embeddings_dir / "items.parquet")
    item_ids = item_metadata["item_id_raw"].astype(str).to_numpy()
    index = ExactIndex(vectors, item_ids)
    seen = (
        pd.concat([train, validation])
        .groupby("user_id")["item_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    positives = (
        test.loc[test["label"].eq(1)]
        .groupby("user_id")["item_id"]
        .apply(lambda x: set(x.astype(str)))
    )
    representatives = (
        test.sort_values("timestamp").drop_duplicates("user_id", keep="last").set_index("user_id")
    )
    maximum_k = max(config.evaluation.ks)
    recommendations: list[list[str]] = []
    per_k: dict[int, list[dict[str, float]]] = {k: [] for k in config.evaluation.ks}
    per_user: dict[str, dict[str, float]] = {}
    latencies: list[float] = []
    for user_id, relevant in positives.items():
        if user_id not in representatives.index:
            continue
        row = representatives.loc[[user_id]].copy()
        with torch.inference_mode():
            query = model.user_tower(move_batch(user_frame_to_batch(row), device)).cpu().numpy()
        started = time.perf_counter()
        _, candidates = index.search(query, len(item_ids))
        latencies.append((time.perf_counter() - started) * 1000)
        filtered = [item for item in candidates[0] if item not in seen.get(user_id, set())][
            :maximum_k
        ]
        recommendations.append(filtered)
        for k in config.evaluation.ks:
            user_metrics = ranking_metrics(filtered, relevant, k)
            per_k[k].append(user_metrics)
            if k == maximum_k:
                per_user[str(user_id)] = user_metrics
    popularity = dict(zip(item_ids, item_metadata["popularity_raw"].astype(float), strict=True))
    categories = dict(zip(item_ids, item_metadata["category_raw"].astype(str), strict=True))
    metrics = {
        f"{name}@{k}": value
        for k, rows in per_k.items()
        for name, value in _aggregate(rows).items()
    }
    metrics.update(
        {
            "catalog_coverage": catalog_coverage(recommendations, set(item_ids)),
            "user_coverage": len(recommendations) / max(len(positives), 1),
            "novelty": novelty(recommendations, popularity),
            "category_diversity": category_diversity(recommendations, categories),
            "retrieval_latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "retrieval_latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        }
    )
    recommended_flat = [item for row in recommendations for item in row]
    head_cutoff = set(
        item_metadata.nlargest(max(1, len(item_metadata) // 10), "popularity_raw")["item_id_raw"]
        .astype(str)
        .tolist()
    )
    metrics["mean_recommended_popularity"] = (
        float(np.mean([popularity[item] for item in recommended_flat])) if recommended_flat else 0.0
    )
    metrics["head_item_share"] = (
        sum(item in head_cutoff for item in recommended_flat) / len(recommended_flat)
        if recommended_flat
        else 0.0
    )

    raw_users = pd.read_parquet(root / "datasets" / dataset_version / "users.parquet").set_index(
        "user_id"
    )
    train_activity = train.groupby("user_id").size().to_dict()
    segment_rows: dict[str, dict[str, list[dict[str, float]]]] = {
        "country": {},
        "subscription_tier": {},
        "device": {},
        "user_lifecycle": {},
        "activity": {},
    }
    train_users = set(train["user_id"].astype(str))
    for user_id, user_metrics in per_user.items():
        user = raw_users.loc[user_id]
        device_label = str(representatives.loc[user_id]["device"])
        count = int(train_activity.get(user_id, 0))
        labels: dict[str, str] = {
            "country": str(user["country"]),
            "subscription_tier": str(user["subscription_tier"]),
            "device": device_label,
            "user_lifecycle": "existing" if user_id in train_users else "new",
            "activity": "new" if count == 0 else "sparse" if count <= 5 else "active",
        }
        for dimension, value in labels.items():
            segment_rows[dimension].setdefault(value, []).append(user_metrics)
    segments = {
        dimension: {
            value: {"user_count": len(rows), **_aggregate(rows)} for value, rows in values.items()
        }
        for dimension, values in segment_rows.items()
    }

    confidence_intervals: dict[str, dict[str, float]] = {}
    if config.evaluation.bootstrap_samples and per_user:
        rng = np.random.default_rng(config.seed)
        rows = list(per_user.values())
        for metric_name in ("recall", "ndcg", "mrr"):
            estimates = [
                float(
                    np.mean(
                        [
                            rows[index][metric_name]
                            for index in rng.integers(0, len(rows), len(rows))
                        ]
                    )
                )
                for _ in range(config.evaluation.bootstrap_samples)
            ]
            confidence_intervals[f"{metric_name}@{maximum_k}"] = {
                "lower_95": float(np.percentile(estimates, 2.5)),
                "upper_95": float(np.percentile(estimates, 97.5)),
            }
    popular_order = (
        item_metadata.sort_values("popularity_raw", ascending=False)["item_id_raw"]
        .astype(str)
        .tolist()
    )
    baseline_rows = []
    for user_id, relevant in positives.items():
        baseline = [item for item in popular_order if item not in seen.get(user_id, set())][
            :maximum_k
        ]
        baseline_rows.append(ranking_metrics(baseline, relevant, maximum_k))
    report = {
        "model_version": model_version,
        "embedding_version": embedding_version,
        "evaluated_users": len(recommendations),
        "metrics": metrics,
        "segments": segments,
        "confidence_intervals": confidence_intervals,
        "popularity_baseline": _aggregate(baseline_rows),
        "assumptions": [
            "test positives are event-time later than train and validation",
            "items seen before the test boundary are excluded",
            "offline results do not establish online business impact",
        ],
    }
    config.paths.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.paths.report_dir / f"evaluation-{model_version}.json"
    atomic_write_json(json_path, report)
    pd.DataFrame(
        [{"metric": name, "value": value} for name, value in sorted(metrics.items())]
    ).to_csv(config.paths.report_dir / f"evaluation-{model_version}.csv", index=False)
    markdown = [
        "# Offline evaluation",
        "",
        f"Model: `{model_version}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    markdown.extend(f"| {name} | {value:.6f} |" for name, value in sorted(metrics.items()))
    markdown.extend(
        [
            "",
            "Offline estimates are biased by historical exposure; "
            "validate changes with online experiments.",
        ]
    )
    (config.paths.report_dir / f"evaluation-{model_version}.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return json_path
