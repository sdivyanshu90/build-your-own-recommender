import json

import pandas as pd
import pytest

from recommender.batch.job import batch_recommend
from recommender.data.preparation import prepare_data
from recommender.data.synthetic import generate_synthetic
from recommender.embeddings.export import export_item_embeddings
from recommender.evaluation.evaluator import evaluate_model
from recommender.features.processor import fit_and_transform_features
from recommender.indexing.build import build_index, load_index
from recommender.retrieval.runtime import RecommendationRuntime
from recommender.training.trainer import train_model


@pytest.mark.integration
def test_data_to_runtime_pipeline(tiny_config) -> None:
    generate_synthetic(tiny_config)
    prepare_data(tiny_config)
    fit_and_transform_features(tiny_config)
    train_model(tiny_config)
    resume_checkpoint = tiny_config.paths.artifact_dir / "models" / "model-v001" / "best.pt"
    resumed_config = tiny_config.model_copy(
        update={
            "training": tiny_config.training.model_copy(
                update={"resume_checkpoint": resume_checkpoint}
            )
        }
    )
    resumed_dir = train_model(resumed_config, model_version="model-resumed-v001")
    resumed_metadata = json.loads((resumed_dir / "training.json").read_text(encoding="utf-8"))
    assert resumed_metadata["resumed_from"] == str(resume_checkpoint.resolve())
    export_item_embeddings(tiny_config)
    build_index(tiny_config)
    index = load_index(tiny_config.paths.artifact_dir / "indexes" / "index-v001", "model-v001")
    assert len(index.item_ids) > 0
    runtime = RecommendationRuntime.load(tiny_config)
    users = pd.read_parquet(
        tiny_config.paths.artifact_dir / "datasets" / "dataset-v001" / "users.parquet"
    )
    result = runtime.recommend(str(users.iloc[0]["user_id"]), 5)
    assert result.candidates
    assert len({candidate.item_id for candidate in result.candidates}) == len(result.candidates)
    unknown = runtime.recommend("missing-user", 5)
    assert unknown.fallback_reason == "unknown_user"
    assert unknown.candidates
    cold = runtime.recommend(
        "cold-with-features",
        3,
        user_features={"country": "IN", "preferred_categories": "books|music"},
        context={"device": "mobile", "position": 2},
    )
    assert cold.fallback_reason in {None, "insufficient_filtered_candidates"}
    constrained = runtime.recommend(
        str(users.iloc[0]["user_id"]),
        2,
        excluded_items={str(index.item_ids[0])},
        category_filter={"books"},
        allow_list=set(map(str, index.item_ids[:10])),
        deny_list={str(index.item_ids[1])},
        maximum_freshness_days=365,
    )
    assert len(constrained.candidates) <= 2
    known_item = str(index.item_ids[0])
    assert runtime.item_embedding(known_item) is not None
    assert runtime.item_embedding("missing-item") is None
    assert runtime.similar_items(known_item, 3).candidates
    assert runtime.similar_items("missing-item", 3).fallback_reason == "unknown_item"
    evaluation_path = evaluate_model(tiny_config)
    assert evaluation_path.exists()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert "segments" in evaluation

    batch_input = tiny_config.paths.data_dir / "batch-users.parquet"
    users[["user_id"]].iloc[:3].to_parquet(batch_input, index=False)
    assert (batch_recommend(tiny_config, batch_input) / "manifest.json").exists()

    faiss_config = tiny_config.model_copy(
        update={"index": tiny_config.index.model_copy(update={"backend": "faiss"})}
    )
    faiss_dir = build_index(faiss_config, index_version="index-faiss-v001")
    faiss_index = load_index(faiss_dir, "model-v001")
    assert faiss_index.search(runtime.embeddings[:1], 3)[1].shape == (1, 3)

    tracked_config = tiny_config.model_copy(
        update={
            "training": tiny_config.training.model_copy(update={"scheduler": "plateau"}),
            "tracking": tiny_config.tracking.model_copy(
                update={
                    "enabled": True,
                    "uri": f"sqlite:///{tiny_config.paths.artifact_dir / 'mlflow.db'}",
                }
            ),
        }
    )
    assert train_model(tracked_config, model_version="model-tracked-v001").exists()
