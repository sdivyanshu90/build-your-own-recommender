import pandas as pd
import pytest

from recommender.data.preparation import clean_events, temporal_split
from recommender.data.synthetic import generate_synthetic
from recommender.data.validation import validate_frames, validate_raw
from recommender.exceptions import DataQualityError


def test_synthetic_generation_is_repeatable(tiny_config, tmp_path) -> None:
    first = generate_synthetic(tiny_config, tmp_path / "one")
    second = generate_synthetic(tiny_config, tmp_path / "two")
    for name in first:
        pd.testing.assert_frame_equal(pd.read_parquet(first[name]), pd.read_parquet(second[name]))


def test_validation_finds_injected_invalid_records(tiny_config, tmp_path) -> None:
    paths = generate_synthetic(tiny_config, tmp_path)
    findings = validate_frames(
        *(pd.read_parquet(paths[name]) for name in ("users", "items", "events"))
    )
    codes = {finding.code for finding in findings}
    assert {
        "null_event_user",
        "invalid_event_type",
        "invalid_position",
        "duplicate_event_id",
    } <= codes


def test_cleaning_removes_invalid_and_duplicate_records(tiny_config, tmp_path) -> None:
    paths = generate_synthetic(tiny_config, tmp_path)
    users, items, events = (pd.read_parquet(paths[name]) for name in ("users", "items", "events"))
    cleaned = clean_events(users, items, events)
    assert cleaned["event_id"].is_unique
    assert cleaned["user_id"].notna().all()
    assert (cleaned["position"] >= 1).all()


def test_temporal_split_has_no_future_leakage() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=100, tz="UTC"), "event_id": range(100)}
    )
    splits = temporal_split(frame, 0.7, 0.15)
    assert splits["train"]["timestamp"].max() < splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() < splits["test"]["timestamp"].min()


def test_validation_reports_missing_columns_and_strict_failure(tiny_config, tmp_path) -> None:
    findings = validate_frames(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert {finding.code for finding in findings} == {"missing_columns"}
    generate_synthetic(tiny_config, tmp_path / "raw")
    with pytest.raises(DataQualityError, match="failed quality"):
        validate_raw(tmp_path / "raw", tmp_path / "report.json", fail_on_error=True)
    with pytest.raises(DataQualityError, match="cannot read"):
        validate_raw(tmp_path / "missing", tmp_path / "missing-report.json")
