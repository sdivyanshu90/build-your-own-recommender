"""Versioned HTTP request and response contracts."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RerankOptions(ApiModel):
    freshness_weight: float = Field(0.0, ge=0, le=10)
    diversity_weight: float = Field(0.15, ge=0, le=1)
    popularity_cap_per_category: int | None = Field(None, ge=1, le=100)


class RecommendationRequest(ApiModel):
    user_id: str = Field(min_length=1, max_length=256)
    top_k: int | None = Field(None, ge=1)
    user_features: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    excluded_item_ids: set[str] = Field(default_factory=set, max_length=5000)
    category_filter: set[str] | None = Field(None, max_length=100)
    allow_list: set[str] | None = Field(None, max_length=5000)
    deny_list: set[str] = Field(default_factory=set, max_length=5000)
    maximum_freshness_days: float | None = Field(None, ge=0)
    request_id: str | None = Field(None, min_length=1, max_length=128)
    experiment_id: str | None = Field(None, max_length=128)
    reranking: RerankOptions | None = None


class SimilarItemsRequest(ApiModel):
    item_id: str = Field(min_length=1, max_length=256)
    top_k: int = Field(10, ge=1)
    excluded_item_ids: set[str] = Field(default_factory=set, max_length=5000)
    request_id: str | None = Field(None, max_length=128)


class BatchRecommendationRequest(ApiModel):
    requests: list[RecommendationRequest] = Field(min_length=1)


class ItemEmbeddingRequest(ApiModel):
    item_id: str = Field(min_length=1, max_length=256)


class RecommendationItem(ApiModel):
    item_id: str
    retrieval_score: float
    final_score: float | None
    rank: int
    reason: str


class RecommendationResponse(ApiModel):
    user_id: str
    request_id: str
    model_version: str
    index_version: str
    recommendations: list[RecommendationItem]
    fallback: bool
    fallback_reason: str | None
    cache_hit: bool
    latency_ms: dict[str, float]


class ErrorResponse(ApiModel):
    error: dict[str, str]
    request_id: str
