import httpx
import pytest

from recommender.data.preparation import prepare_data
from recommender.data.synthetic import generate_synthetic
from recommender.embeddings.export import export_item_embeddings
from recommender.features.processor import fit_and_transform_features
from recommender.indexing.build import build_index
from recommender.retrieval.runtime import RecommendationRuntime
from recommender.serving.app import create_app
from recommender.training.trainer import train_model


@pytest.mark.e2e
async def test_end_to_end_api_returns_fallback_and_versions(tiny_config) -> None:
    generate_synthetic(tiny_config)
    prepare_data(tiny_config)
    fit_and_transform_features(tiny_config)
    train_model(tiny_config)
    export_item_embeddings(tiny_config)
    build_index(tiny_config)
    RecommendationRuntime.load(tiny_config)
    app = create_app(tiny_config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/recommendations", json={"user_id": "brand-new-user", "top_k": 5}
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["fallback"] is True
        assert payload["fallback_reason"] == "unknown_user"
        assert payload["model_version"] == "model-v001"
        assert payload["index_version"] == "index-v001"
        assert len(payload["recommendations"]) == 5
