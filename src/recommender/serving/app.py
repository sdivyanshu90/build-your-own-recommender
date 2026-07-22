"""FastAPI application factory with lifecycle-safe resources."""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, cast

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from recommender import __version__
from recommender.config.models import AppConfig
from recommender.exceptions import ArtifactError, RecommenderError
from recommender.observability.metrics import (
    CACHE_HITS,
    FALLBACKS,
    LATENCY,
    READY,
    REQUESTS,
    RETURNED_ITEMS,
)
from recommender.reranking.policies import RerankConfig
from recommender.retrieval.runtime import RecommendationRuntime
from recommender.serving.schemas import (
    BatchRecommendationRequest,
    ErrorResponse,
    ItemEmbeddingRequest,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    SimilarItemsRequest,
)

MAX_REQUEST_BYTES = 1_048_576


def create_app(config: AppConfig, runtime: RecommendationRuntime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            app.state.runtime = runtime
        else:
            app.state.runtime = RecommendationRuntime.load(config)
        READY.set(1)
        yield
        READY.set(0)
        app.state.runtime.cache.clear()

    app = FastAPI(
        title="Two-Tower Recommender", version=__version__, lifespan=lifespan, debug=False
    )
    if config.serving.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.serving.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Request-ID"],
        )

    @app.middleware("http")
    async def safety_and_correlation(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id[:128]
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "payload_too_large",
                        "message": "request payload exceeds limit",
                    },
                    "request_id": request.state.request_id,
                },
            )
        started = time.perf_counter()
        response = cast(Response, await call_next(request))
        response.headers["X-Request-ID"] = request.state.request_id
        LATENCY.labels(request.url.path).observe(time.perf_counter() - started)
        REQUESTS.labels(request.url.path, str(response.status_code)).inc()
        return response

    @app.exception_handler(RecommenderError)
    async def recommender_error(request: Request, error: RecommenderError) -> JSONResponse:
        return JSONResponse(
            status_code=503 if isinstance(error, ArtifactError) else 400,
            content={
                "error": {"code": error.__class__.__name__, "message": str(error)},
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        del error
        return JSONResponse(
            status_code=500,
            content={
                "error": {"code": "internal_error", "message": "internal service error"},
                "request_id": request.state.request_id,
            },
        )

    def get_runtime(request: Request) -> RecommendationRuntime:
        loaded = getattr(request.app.state, "runtime", None)
        if loaded is None:
            raise HTTPException(503, "runtime is not ready")
        return cast(RecommendationRuntime, loaded)

    def response_for(payload: RecommendationRequest, request: Request) -> RecommendationResponse:
        loaded = get_runtime(request)
        top_k = payload.top_k or config.serving.default_k
        if top_k > config.serving.max_k:
            raise HTTPException(422, f"top_k cannot exceed {config.serving.max_k}")
        options = payload.reranking
        result = loaded.recommend(
            payload.user_id,
            top_k,
            payload.user_features,
            payload.context,
            payload.excluded_item_ids,
            payload.category_filter,
            payload.allow_list,
            payload.deny_list,
            payload.maximum_freshness_days,
            RerankConfig(**options.model_dump()) if options else None,
        )
        if result.fallback_reason:
            FALLBACKS.labels(result.fallback_reason).inc()
        if result.cache_hit:
            CACHE_HITS.inc()
        RETURNED_ITEMS.observe(len(result.candidates))
        return RecommendationResponse(
            user_id=payload.user_id,
            request_id=payload.request_id or request.state.request_id,
            model_version=loaded.model_version,
            index_version=loaded.index_version,
            recommendations=[
                RecommendationItem(
                    item_id=candidate.item_id,
                    retrieval_score=candidate.retrieval_score,
                    final_score=candidate.final_score,
                    rank=rank,
                    reason="retrieval" if candidate.retrieval_score else "fallback",
                )
                for rank, candidate in enumerate(result.candidates, 1)
            ],
            fallback=result.fallback_reason is not None,
            fallback_reason=result.fallback_reason,
            cache_hit=result.cache_hit,
            latency_ms=result.latency_ms,
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request) -> dict[str, str]:
        get_runtime(request)
        return {"status": "ready"}

    @app.get("/version")
    async def version(request: Request) -> dict[str, str]:
        loaded = get_runtime(request)
        return {
            "service": __version__,
            "model": loaded.model_version,
            "index": loaded.index_version,
        }

    @app.get("/v1/model-info")
    async def model_info(request: Request) -> dict[str, Any]:
        loaded = get_runtime(request)
        return {
            "model_version": loaded.model_version,
            "index_version": loaded.index_version,
            "embedding_dimension": config.model.embedding_dim,
            "similarity": config.model.similarity,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/recommendations",
        response_model=RecommendationResponse,
        responses={400: {"model": ErrorResponse}},
    )
    async def recommendations(
        payload: RecommendationRequest, request: Request
    ) -> RecommendationResponse:
        try:
            with anyio.fail_after(config.serving.request_timeout_seconds):
                return await anyio.to_thread.run_sync(partial(response_for, payload, request))
        except TimeoutError as error:
            raise HTTPException(504, "recommendation timed out") from error

    @app.post("/v1/batch-recommendations", response_model=list[RecommendationResponse])
    async def batch_recommendations(
        payload: BatchRecommendationRequest, request: Request
    ) -> list[RecommendationResponse]:
        if len(payload.requests) > config.serving.max_batch_size:
            raise HTTPException(422, f"batch cannot exceed {config.serving.max_batch_size}")
        try:
            with anyio.fail_after(config.serving.request_timeout_seconds * len(payload.requests)):
                return await anyio.to_thread.run_sync(
                    lambda: [response_for(entry, request) for entry in payload.requests]
                )
        except TimeoutError as error:
            raise HTTPException(504, "batch recommendation timed out") from error

    @app.post("/v1/similar-items")
    async def similar_items(payload: SimilarItemsRequest, request: Request) -> dict[str, Any]:
        if payload.top_k > config.serving.max_k:
            raise HTTPException(422, f"top_k cannot exceed {config.serving.max_k}")
        loaded = get_runtime(request)
        result = loaded.similar_items(payload.item_id, payload.top_k, payload.excluded_item_ids)
        return {
            "item_id": payload.item_id,
            "request_id": payload.request_id or request.state.request_id,
            "model_version": loaded.model_version,
            "index_version": loaded.index_version,
            "recommendations": [
                {"item_id": item.item_id, "retrieval_score": item.retrieval_score, "rank": rank}
                for rank, item in enumerate(result.candidates, 1)
            ],
            "fallback_reason": result.fallback_reason,
            "latency_ms": result.latency_ms,
        }

    @app.post("/v1/item-embedding")
    async def item_embedding(payload: ItemEmbeddingRequest, request: Request) -> dict[str, Any]:
        loaded = get_runtime(request)
        embedding = loaded.item_embedding(payload.item_id)
        if embedding is None:
            raise HTTPException(404, "item not found")
        return {
            "item_id": payload.item_id,
            "model_version": loaded.model_version,
            "embedding": embedding,
        }

    return app
