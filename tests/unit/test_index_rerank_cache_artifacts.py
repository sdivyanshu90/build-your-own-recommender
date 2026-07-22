import numpy as np
import pytest

from recommender.artifacts.manifest import ArtifactManifest
from recommender.caching.memory import InMemoryCache
from recommender.exceptions import ArtifactError, CompatibilityError
from recommender.indexing.exact import ExactIndex
from recommender.reranking.policies import Candidate, RerankConfig, apply_policies


def test_exact_index_orders_inner_products() -> None:
    index = ExactIndex(np.eye(3, dtype=np.float32), np.array(["a", "b", "c"]))
    scores, items = index.search(np.array([[0.1, 0.9, 0]], dtype=np.float32), 2)
    assert items.tolist() == [["b", "a"]]
    assert scores[0, 0] > scores[0, 1]
    empty_scores, empty_items = index.search(np.ones((2, 3), dtype=np.float32), 0)
    assert empty_scores.shape == empty_items.shape == (2, 0)
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones((1, 2), dtype=np.float32), 1)


def test_exact_index_rejects_misaligned_data() -> None:
    with pytest.raises(ValueError, match="aligned"):
        ExactIndex(np.ones((2, 3)), np.array(["one"]))


def test_reranker_filters_deduplicates_and_diversifies() -> None:
    candidates = [
        Candidate("a", 1.0, "x", 1, 10, True),
        Candidate("a", 0.9, "x", 1, 10, True),
        Candidate("b", 0.95, "x", 1, 10, True),
        Candidate("c", 0.9, "y", 1, 1, True),
        Candidate("d", 2.0, "z", 1, 1, False),
    ]
    result = apply_policies(candidates, 3, RerankConfig(diversity_weight=0.2))
    assert [entry.item_id for entry in result] == ["a", "c", "b"]


def test_cache_expiration(monkeypatch) -> None:
    now = [1.0]
    monkeypatch.setattr("recommender.caching.memory.time.monotonic", lambda: now[0])
    cache = InMemoryCache[int](1)
    cache.set("a", 1, 2)
    assert cache.get("a") == 1
    now[0] = 4
    assert cache.get("a") is None
    cache.set("a", 1, 2)
    cache.set("b", 2, 2)
    assert cache.get("a") is None
    cache.clear()
    assert cache.get("b") is None


def test_cache_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="positive"):
        InMemoryCache(0)


def test_manifest_detects_tampering_and_incompatibility(tmp_path) -> None:
    directory = tmp_path / "artifact"
    directory.mkdir()
    (directory / "data.txt").write_text("safe", encoding="utf-8")
    manifest = ArtifactManifest.create("model", "model-v001", {}, {}, {"features": "features-v001"})
    manifest.write(directory)
    assert (directory / "manifest.json").stat().st_mode & 0o777 == 0o644
    loaded = ArtifactManifest.load(directory)
    with pytest.raises(CompatibilityError):
        loaded.require_dependency("features", "features-v002")
    (directory / "data.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactError, match="checksum"):
        ArtifactManifest.load(directory)
