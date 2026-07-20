import httpx
import pytest

from recommender.caching.memory import InMemoryCache
from recommender.reranking.policies import Candidate
from recommender.retrieval.runtime import RetrievalResult
from recommender.serving.app import MAX_REQUEST_BYTES, create_app


class FakeRuntime:
    model_version = "model-test"
    index_version = "index-test"

    def __init__(self) -> None:
        self.cache = InMemoryCache()

    def recommend(self, user_id, top_k, *args, **kwargs):
        return RetrievalResult(
            [
                Candidate(f"item-{index}", 1 / (index + 1), "books", 0.1, 1, True, 1 / (index + 1))
                for index in range(top_k)
            ],
            "unknown_user" if user_id == "unknown" else None,
            {"total": 1.0},
        )

    def similar_items(self, item_id, top_k, excluded):
        return self.recommend(item_id, top_k)

    def item_embedding(self, item_id):
        return [1.0, 0.0] if item_id == "known" else None


@pytest.fixture
async def client(tiny_config):
    app = create_app(tiny_config, FakeRuntime())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as value:
            yield value


async def test_recommendation_contract(client) -> None:
    response = await client.post(
        "/v1/recommendations", json={"user_id": "u1", "top_k": 3, "request_id": "r1"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "r1"
    assert payload["model_version"] == "model-test"
    assert [item["rank"] for item in payload["recommendations"]] == [1, 2, 3]
    assert response.headers["X-Request-ID"]


async def test_unknown_fields_and_excessive_k_are_rejected(client) -> None:
    assert (
        await client.post("/v1/recommendations", json={"user_id": "u", "surprise": True})
    ).status_code == 422
    assert (
        await client.post("/v1/recommendations", json={"user_id": "u", "top_k": 21})
    ).status_code == 422
    too_large_batch = await client.post(
        "/v1/batch-recommendations",
        json={"requests": [{"user_id": str(index)} for index in range(6)]},
    )
    assert too_large_batch.status_code == 422
    assert (
        await client.post("/v1/similar-items", json={"item_id": "known", "top_k": 21})
    ).status_code == 422


async def test_health_metrics_and_embedding_contracts(client) -> None:
    assert (await client.get("/health/live")).json() == {"status": "alive"}
    assert (await client.get("/health/ready")).json() == {"status": "ready"}
    assert "recommender_ready" in (await client.get("/metrics")).text
    assert (await client.post("/v1/item-embedding", json={"item_id": "known"})).json()[
        "embedding"
    ] == [1.0, 0.0]
    assert (await client.post("/v1/item-embedding", json={"item_id": "missing"})).status_code == 404
    assert (await client.get("/version")).json()["model"] == "model-test"
    assert (await client.get("/v1/model-info")).json()["index_version"] == "index-test"
    similar = await client.post("/v1/similar-items", json={"item_id": "known", "top_k": 2})
    assert similar.status_code == 200
    assert len(similar.json()["recommendations"]) == 2
    batch = await client.post(
        "/v1/batch-recommendations",
        json={"requests": [{"user_id": "u1"}, {"user_id": "unknown"}]},
    )
    assert batch.status_code == 200
    assert len(batch.json()) == 2


async def test_oversized_payload_is_rejected(client) -> None:
    response = await client.post(
        "/v1/recommendations",
        content=b"{}",
        headers={"content-length": str(MAX_REQUEST_BYTES + 1), "content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
