"""Immutable, thread-safe recommendation runtime bundle."""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from recommender.artifacts.manifest import ArtifactManifest
from recommender.caching.memory import InMemoryCache
from recommender.config.models import AppConfig
from recommender.features.processor import FeatureProcessor
from recommender.indexing.base import VectorIndex
from recommender.indexing.build import load_index
from recommender.models.two_tower import TwoTowerModel
from recommender.reranking.policies import Candidate, RerankConfig, apply_policies
from recommender.training.datasets import move_batch, user_frame_to_batch
from recommender.training.trainer import load_model


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[Candidate]
    fallback_reason: str | None
    latency_ms: dict[str, float]
    cache_hit: bool = False


class RecommendationRuntime:
    """Loaded compatible model/index pair; all request operations are read-only."""

    def __init__(
        self,
        config: AppConfig,
        processor: FeatureProcessor,
        model: TwoTowerModel,
        index: VectorIndex,
        users: pd.DataFrame,
        items: pd.DataFrame,
        seen: dict[str, set[str]],
        embedding_ids: np.ndarray,
        embeddings: np.ndarray,
        model_version: str,
        index_version: str,
    ) -> None:
        self.config = config
        self.processor = processor
        self.model = model
        self.index = index
        self.users = users.set_index("user_id", drop=False)
        self.items = items.set_index("item_id_raw", drop=False)
        self.seen = seen
        self.model_version = model_version
        self.index_version = index_version
        self.device = next(model.parameters()).device
        self.embedding_position = {
            str(item): position for position, item in enumerate(embedding_ids)
        }
        self.embeddings = embeddings
        self.cache: InMemoryCache[RetrievalResult] = InMemoryCache(max_entries=5000)

    @classmethod
    def load(
        cls,
        config: AppConfig,
        dataset_version: str = "dataset-v001",
        feature_version: str = "features-v001",
        model_version: str = "model-v001",
        embedding_version: str = "embeddings-v001",
        index_version: str = "index-v001",
    ) -> "RecommendationRuntime":
        root = config.paths.artifact_dir
        processor = FeatureProcessor.load(root / "feature-pipelines" / feature_version)
        model = load_model(root / "models" / model_version, processor, config)
        index = load_index(root / "indexes" / index_version, model_version)
        embedding_dir = root / "embeddings" / embedding_version
        embedding_manifest = ArtifactManifest.load(embedding_dir)
        embedding_manifest.require_dependency("model", model_version)
        embedding_ids = (
            pd.read_parquet(embedding_dir / "items.parquet")["item_id_raw"].astype(str).to_numpy()
        )
        embeddings = np.load(embedding_dir / "item_embeddings.npy", allow_pickle=False)
        dataset = root / "datasets" / dataset_version
        users = pd.read_parquet(dataset / "users.parquet")
        items = pd.read_parquet(embedding_dir / "items.parquet")
        event_frames = [
            pd.read_parquet(dataset / f"{split}.parquet")
            for split in ("train", "validation", "test")
        ]
        seen: dict[str, set[str]] = {}
        for user_id, values in pd.concat(event_frames).groupby("user_id")["item_id"]:
            seen[str(user_id)] = set(values.astype(str))
        return cls(
            config,
            processor,
            model,
            index,
            users,
            items,
            seen,
            embedding_ids,
            embeddings,
            model_version,
            index_version,
        )

    def _user_embedding(
        self, user_id: str, user_features: dict[str, Any] | None, context: dict[str, Any] | None
    ) -> np.ndarray | None:
        if user_id in self.users.index:
            raw = self.users.loc[[user_id]].copy()
        elif user_features:
            defaults: dict[str, Any] = {
                "user_id": user_id,
                "age_bucket": "<UNK>",
                "country": "<UNK>",
                "language": "<UNK>",
                "subscription_tier": "free",
                "device_preference": "mobile",
                "account_age_days": 0,
                "preferred_categories": "",
                "activity_score": 0.0,
            }
            defaults.update(user_features)
            raw = pd.DataFrame([defaults])
        else:
            return None
        if user_features:
            for key, value in user_features.items():
                if key in raw.columns and key != "user_id":
                    raw.loc[:, key] = value
        transformed = self.processor.transform_users(raw)
        device_value = (context or {}).get("device", raw.iloc[0]["device_preference"])
        position = float((context or {}).get("position", 1))
        transformed["context_device_idx"] = self.processor.encode("context.device", device_value)
        transformed["context_position_z"] = np.log1p(max(position, 1)) / np.log(51)
        with torch.inference_mode():
            embedding = self.model.user_tower(
                move_batch(user_frame_to_batch(transformed), self.device)
            )
            return np.asarray(embedding.cpu().numpy(), dtype=np.float32)

    def _eligible(
        self,
        item_id: str,
        excluded: set[str],
        category_filter: set[str] | None,
        allow_list: set[str] | None,
        deny_list: set[str],
        maximum_freshness_days: float | None,
    ) -> bool:
        if item_id in excluded or item_id in deny_list or item_id not in self.items.index:
            return False
        if allow_list is not None and item_id not in allow_list:
            return False
        item = self.items.loc[item_id]
        if not bool(item["available"]):
            return False
        if category_filter and str(item["category_raw"]) not in category_filter:
            return False
        return not (
            maximum_freshness_days is not None
            and float(item["freshness_days_raw"]) > maximum_freshness_days
        )

    def _fallback(
        self,
        limit: int,
        excluded: set[str],
        category_filter: set[str] | None,
        allow_list: set[str] | None,
        deny_list: set[str],
        maximum_freshness_days: float | None,
    ) -> list[Candidate]:
        ordered = self.items.sort_values(
            ["popularity_raw", "freshness_days_raw"], ascending=[False, True]
        )
        result: list[Candidate] = []
        for item_id, item in ordered.iterrows():
            if self._eligible(
                str(item_id),
                excluded,
                category_filter,
                allow_list,
                deny_list,
                maximum_freshness_days,
            ):
                result.append(
                    Candidate(
                        str(item_id),
                        0.0,
                        str(item["category_raw"]),
                        float(item["popularity_raw"]),
                        float(item["freshness_days_raw"]),
                        True,
                        0.0,
                    )
                )
            if len(result) >= limit:
                break
        return result

    def recommend(
        self,
        user_id: str,
        top_k: int,
        user_features: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        excluded_items: set[str] | None = None,
        category_filter: set[str] | None = None,
        allow_list: set[str] | None = None,
        deny_list: set[str] | None = None,
        maximum_freshness_days: float | None = None,
        rerank: RerankConfig | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        excluded = set(excluded_items or ()) | self.seen.get(user_id, set())
        denied = set(deny_list or ())
        cache_payload = {
            "u": user_id,
            "k": top_k,
            "uf": user_features,
            "c": context,
            "x": sorted(excluded),
            "cat": sorted(category_filter or ()),
            "allow": sorted(allow_list) if allow_list is not None else None,
            "deny": sorted(denied),
            "fresh": maximum_freshness_days,
            "rr": (rerank or RerankConfig()).__dict__,
            "m": self.model_version,
            "i": self.index_version,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return RetrievalResult(
                cached.candidates, cached.fallback_reason, cached.latency_ms, True
            )
        tower_started = time.perf_counter()
        query = self._user_embedding(user_id, user_features, context)
        tower_ms = (time.perf_counter() - tower_started) * 1000
        if query is None:
            candidates = self._fallback(
                top_k, excluded, category_filter, allow_list, denied, maximum_freshness_days
            )
            result = RetrievalResult(
                candidates,
                "unknown_user",
                {
                    "user_tower": tower_ms,
                    "ann": 0.0,
                    "rerank": 0.0,
                    "total": (time.perf_counter() - started) * 1000,
                },
            )
            self.cache.set(cache_key, result, 30)
            return result
        ann_started = time.perf_counter()
        search_k = min(
            len(self.index.item_ids), max(self.config.index.search_candidates, top_k * 5)
        )
        scores, item_ids = self.index.search(query, search_k)
        ann_ms = (time.perf_counter() - ann_started) * 1000
        candidates = []
        for score, item_id in zip(scores[0], item_ids[0], strict=True):
            item_key = str(item_id)
            if self._eligible(
                item_key, excluded, category_filter, allow_list, denied, maximum_freshness_days
            ):
                item = self.items.loc[item_key]
                candidates.append(
                    Candidate(
                        item_key,
                        float(score),
                        str(item["category_raw"]),
                        float(item["popularity_raw"]),
                        float(item["freshness_days_raw"]),
                        bool(item["available"]),
                    )
                )
        rerank_started = time.perf_counter()
        ranked = apply_policies(candidates, top_k, rerank or RerankConfig())
        fallback_reason = None
        if len(ranked) < top_k:
            existing = excluded | {candidate.item_id for candidate in ranked}
            ranked.extend(
                self._fallback(
                    top_k - len(ranked),
                    existing,
                    category_filter,
                    allow_list,
                    denied,
                    maximum_freshness_days,
                )
            )
            fallback_reason = "insufficient_filtered_candidates"
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        result = RetrievalResult(
            ranked,
            fallback_reason,
            {
                "user_tower": tower_ms,
                "ann": ann_ms,
                "rerank": rerank_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        )
        self.cache.set(cache_key, result, 30)
        return result

    def similar_items(
        self, item_id: str, top_k: int, excluded: set[str] | None = None
    ) -> RetrievalResult:
        started = time.perf_counter()
        position = self.embedding_position.get(item_id)
        if position is None:
            return RetrievalResult(
                self._fallback(top_k, set(), None, None, set(), None),
                "unknown_item",
                {"ann": 0.0, "total": 0.0},
            )
        ann_started = time.perf_counter()
        scores, ids = self.index.search(
            self.embeddings[position : position + 1], min(len(self.index.item_ids), top_k + 20)
        )
        ann_ms = (time.perf_counter() - ann_started) * 1000
        candidates = []
        denied = {item_id} | set(excluded or ())
        for score, candidate_id in zip(scores[0], ids[0], strict=True):
            key = str(candidate_id)
            if self._eligible(key, denied, None, None, set(), None):
                item = self.items.loc[key]
                candidates.append(
                    Candidate(
                        key,
                        float(score),
                        str(item["category_raw"]),
                        float(item["popularity_raw"]),
                        float(item["freshness_days_raw"]),
                        True,
                        float(score),
                    )
                )
            if len(candidates) >= top_k:
                break
        return RetrievalResult(
            candidates, None, {"ann": ann_ms, "total": (time.perf_counter() - started) * 1000}
        )

    def item_embedding(self, item_id: str) -> list[float] | None:
        position = self.embedding_position.get(item_id)
        return self.embeddings[position].astype(float).tolist() if position is not None else None
