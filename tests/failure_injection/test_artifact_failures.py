import pytest

from recommender.artifacts.manifest import ArtifactManifest
from recommender.exceptions import ArtifactError, CompatibilityError
from recommender.indexing.build import load_index


def test_missing_index_manifest_fails_closed(tmp_path) -> None:
    with pytest.raises(ArtifactError, match="invalid artifact manifest"):
        load_index(tmp_path / "missing")


def test_malformed_manifest_fails_closed(tmp_path) -> None:
    directory = tmp_path / "artifact"
    directory.mkdir()
    (directory / "manifest.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="invalid artifact manifest"):
        ArtifactManifest.load(directory)


def test_model_index_mismatch_fails_before_loading_vectors(tmp_path) -> None:
    directory = tmp_path / "index"
    manifest = ArtifactManifest.create(
        "index",
        "index-v001",
        {},
        {"dimension": 4},
        dependencies={"model": "model-v001", "embeddings": "embeddings-v001"},
        metadata={"backend": "exact", "dimension": 4},
    )
    manifest.write(directory)
    with pytest.raises(CompatibilityError, match="requires model"):
        load_index(directory, "model-v002")
