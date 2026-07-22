"""Strict hierarchical configuration models."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(StrictModel):
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")


class DataConfig(StrictModel):
    num_users: int = Field(1000, ge=10)
    num_items: int = Field(2000, ge=20)
    num_events: int = Field(50000, ge=100)
    invalid_fraction: float = Field(0.01, ge=0, lt=0.5)
    duplicate_fraction: float = Field(0.01, ge=0, lt=0.5)
    train_fraction: float = Field(0.8, gt=0.5, lt=0.95)
    validation_fraction: float = Field(0.1, gt=0, lt=0.4)
    positive_events: tuple[str, ...] = ("click", "view", "add_to_cart", "purchase", "rating")
    positive_weight_threshold: float = Field(1.0, ge=0)

    @model_validator(mode="after")
    def validate_splits(self) -> "DataConfig":
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
        return self


class FeatureConfig(StrictModel):
    min_frequency: int = Field(1, ge=1)
    max_history: int = Field(50, ge=1, le=1000)


class ModelConfig(StrictModel):
    embedding_dim: int = Field(64, ge=4, le=2048)
    id_embedding_dim: int = Field(32, ge=2)
    categorical_embedding_dim: int = Field(8, ge=2)
    hidden_dims: tuple[int, ...] = (128, 64)
    activation: Literal["relu", "gelu", "silu"] = "gelu"
    dropout: float = Field(0.1, ge=0, lt=1)
    similarity: Literal["cosine", "dot"] = "cosine"
    temperature: float = Field(0.07, gt=0, le=1)


class TrainingConfig(StrictModel):
    epochs: int = Field(10, ge=1)
    batch_size: int = Field(256, ge=2)
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-5, ge=0)
    patience: int = Field(3, ge=1)
    gradient_clip_norm: float = Field(5.0, gt=0)
    num_workers: int = Field(0, ge=0)
    deterministic: bool = True
    mixed_precision: bool = True
    negative_strategy: Literal["in_batch", "uniform", "popularity", "hard"] = "in_batch"
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    symmetric_loss: bool = False
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    resume_checkpoint: Path | None = None


class EvaluationConfig(StrictModel):
    ks: tuple[int, ...] = (5, 10, 20)
    bootstrap_samples: int = Field(200, ge=0)

    @model_validator(mode="after")
    def validate_ks(self) -> "EvaluationConfig":
        if not self.ks or any(k < 1 for k in self.ks) or len(set(self.ks)) != len(self.ks):
            raise ValueError("ks must contain unique positive integers")
        return self


class IndexConfig(StrictModel):
    backend: Literal["exact", "faiss"] = "exact"
    metric: Literal["cosine", "dot"] = "cosine"
    search_candidates: int = Field(200, ge=1)
    hnsw_m: int = Field(32, ge=4, le=128)
    hnsw_ef_construction: int = Field(200, ge=8)
    hnsw_ef_search: int = Field(128, ge=8)


class ServingConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    max_k: int = Field(100, ge=1, le=1000)
    default_k: int = Field(10, ge=1)
    max_batch_size: int = Field(100, ge=1, le=10000)
    request_timeout_seconds: float = Field(2.0, gt=0, le=30)
    debug: bool = False
    cors_origins: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_limits(self) -> "ServingConfig":
        if self.default_k > self.max_k:
            raise ValueError("default_k cannot exceed max_k")
        return self


class TrackingConfig(StrictModel):
    enabled: bool = False
    uri: str = "sqlite:///./artifacts/mlflow.db"
    experiment: str = "two-tower"


class AppConfig(StrictModel):
    environment: Literal["development", "test", "production"] = "development"
    seed: int = Field(1729, ge=0, le=2**32 - 1)
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    index: IndexConfig = IndexConfig()
    serving: ServingConfig = ServingConfig()
    tracking: TrackingConfig = TrackingConfig()

    @model_validator(mode="after")
    def validate_compatibility(self) -> "AppConfig":
        if self.index.metric != self.model.similarity:
            raise ValueError("index.metric must match model.similarity")
        return self
