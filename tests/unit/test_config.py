from pathlib import Path

import pytest

from recommender.config.loading import load_config
from recommender.config.models import AppConfig
from recommender.exceptions import ConfigurationError
from recommender.training.trainer import train_model


def test_rejects_unknown_configuration() -> None:
    with pytest.raises(ValueError, match="extra"):
        AppConfig.model_validate({"unexpected": True})


def test_rejects_model_index_metric_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        AppConfig.model_validate({"model": {"similarity": "cosine"}, "index": {"metric": "dot"}})


def test_environment_override_is_nested(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    config = load_config(path, {"RECOMMENDER__TRAINING__BATCH_SIZE": "64"})
    assert config.seed == 1
    assert config.training.batch_size == 64


def test_missing_configuration_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml_and_non_mapping_are_actionable(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("key: [", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_config(invalid)
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("hello", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_config(scalar)


def test_environment_string_and_top_level_override(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("environment: development\n", encoding="utf-8")
    config = load_config(path, {"RECOMMENDER__ENVIRONMENT": "production", "IGNORED": "value"})
    assert config.environment == "production"


def test_trainer_rejects_unwired_negative_strategy(tiny_config: AppConfig) -> None:
    sampled = tiny_config.model_copy(
        update={
            "training": tiny_config.training.model_copy(update={"negative_strategy": "uniform"})
        }
    )
    with pytest.raises(ConfigurationError, match="supports negative_strategy='in_batch' only"):
        train_model(sampled)
