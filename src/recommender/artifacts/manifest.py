"""Versioned artifact manifest contracts."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from recommender.exceptions import ArtifactError, CompatibilityError
from recommender.utils.hashing import canonical_hash, file_sha256
from recommender.utils.io import atomic_write_json


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["dataset", "features", "model", "embeddings", "index", "batch"]
    version: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]+$")
    created_at: datetime
    config_hash: str
    schema_hash: str
    dependencies: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        artifact_type: Literal["dataset", "features", "model", "embeddings", "index", "batch"],
        version: str,
        config: Any,
        schema: Any,
        dependencies: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ArtifactManifest":
        return cls(
            artifact_type=artifact_type,
            version=version,
            created_at=datetime.now(UTC),
            config_hash=canonical_hash(config),
            schema_hash=canonical_hash(schema),
            dependencies=dependencies or {},
            metadata=metadata or {},
        )

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        files = {
            path.relative_to(directory).as_posix(): file_sha256(path)
            for path in directory.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        updated = self.model_copy(update={"files": files})
        atomic_write_json(directory / "manifest.json", updated.model_dump(mode="json"))

    @classmethod
    def load(cls, directory: Path, verify: bool = True) -> "ArtifactManifest":
        path = directory / "manifest.json"
        try:
            manifest = cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ArtifactError(f"invalid artifact manifest: {path}") from error
        if verify:
            for relative, checksum in manifest.files.items():
                candidate = directory / relative
                if not candidate.is_file() or file_sha256(candidate) != checksum:
                    raise ArtifactError(f"artifact checksum mismatch: {relative}")
        return manifest

    def require_dependency(self, name: str, version: str) -> None:
        actual = self.dependencies.get(name)
        if actual != version:
            raise CompatibilityError(
                f"{self.artifact_type} {self.version} requires {name}={actual!r}, got {version!r}"
            )
