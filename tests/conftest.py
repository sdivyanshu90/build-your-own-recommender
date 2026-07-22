"""Deterministic shared test fixtures."""

import os
from pathlib import Path
from typing import Any

import pytest

from recommender.config.models import AppConfig

FAILED_REPORTS: list[Any] = []


def pytest_runtest_logreport(report: Any) -> None:
    """Retain failed reports until pytest has suspended output capture."""
    if report.failed:
        FAILED_REPORTS.append(report)


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: Any,
) -> None:
    """Publish actionable first-party GitHub annotations after capture is suspended."""
    if os.getenv("GITHUB_ACTIONS") != "true":
        return
    for report in FAILED_REPORTS:
        path, line, test_name = report.location
        details = str(report.longrepr).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        terminalreporter.write_line(
            f"::error file={path},line={line + 1},title=pytest failure: {test_name}::{details}"
        )
    if exitstatus and not FAILED_REPORTS:
        terminalreporter.write_line(
            f"::error title=pytest session failure::pytest exited with status {exitstatus} "
            "without an individual failed test report"
        )


@pytest.fixture
def tiny_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "environment": "test",
            "seed": 7,
            "paths": {
                "data_dir": tmp_path / "data",
                "artifact_dir": tmp_path / "artifacts",
                "report_dir": tmp_path / "reports",
            },
            "data": {
                "num_users": 24,
                "num_items": 48,
                "num_events": 400,
                "invalid_fraction": 0.02,
                "duplicate_fraction": 0.02,
                "train_fraction": 0.7,
                "validation_fraction": 0.15,
            },
            "features": {"min_frequency": 1, "max_history": 10},
            "model": {
                "embedding_dim": 8,
                "id_embedding_dim": 4,
                "categorical_embedding_dim": 3,
                "hidden_dims": [16],
                "dropout": 0.0,
                "similarity": "cosine",
                "temperature": 0.1,
            },
            "training": {
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.003,
                "patience": 1,
                "num_workers": 0,
                "deterministic": True,
            },
            "evaluation": {"ks": [5, 10], "bootstrap_samples": 0},
            "index": {"backend": "exact", "metric": "cosine", "search_candidates": 30},
            "serving": {"max_k": 20, "default_k": 5, "max_batch_size": 5},
        }
    )
