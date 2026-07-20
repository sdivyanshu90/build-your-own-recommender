"""Atomic local artifact-version pointer publication."""

from pathlib import Path

from recommender.artifacts.manifest import ArtifactManifest
from recommender.utils.io import atomic_write_json, ensure_within


def publish_current(artifact_directory: Path, collection_root: Path) -> Path:
    safe = ensure_within(artifact_directory, collection_root)
    manifest = ArtifactManifest.load(safe)
    pointer = collection_root / "current.json"
    atomic_write_json(
        pointer,
        {
            "version": manifest.version,
            "artifact_type": manifest.artifact_type,
            "manifest_checksum": manifest.model_dump(mode="json")["files"],
        },
    )
    return pointer
