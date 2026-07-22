"""YAML loading with nested environment overrides."""

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from recommender.config.models import AppConfig
from recommender.exceptions import ConfigurationError

ENV_PREFIX = "RECOMMENDER__"


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _environment_overrides(environment: dict[str, str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (environment or dict(os.environ)).items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        node = result
        for component in path[:-1]:
            node = node.setdefault(component, {})
        node[path[-1]] = _parse_value(value)
    return result


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path, environment: dict[str, str] | None = None) -> AppConfig:
    """Load and validate configuration, applying RECOMMENDER__ nested overrides."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"configuration file does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML configuration: {path}") from error
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping")
    try:
        return AppConfig.model_validate(_merge(raw, _environment_overrides(environment)))
    except ValidationError as error:
        raise ConfigurationError(str(error)) from error
