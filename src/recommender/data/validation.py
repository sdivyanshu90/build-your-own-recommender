"""Data-quality validation with machine-readable findings."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from recommender.data.schemas import EVENT_COLUMNS, EVENT_WEIGHTS, ITEM_COLUMNS, USER_COLUMNS
from recommender.exceptions import DataQualityError
from recommender.utils.io import atomic_write_json


@dataclass(frozen=True)
class QualityFinding:
    code: str
    severity: str
    count: int
    message: str


def _missing_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> set[str]:
    return set(required).difference(frame.columns)


def validate_frames(
    users: pd.DataFrame, items: pd.DataFrame, events: pd.DataFrame
) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for name, frame, columns in (
        ("users", users, USER_COLUMNS),
        ("items", items, ITEM_COLUMNS),
        ("events", events, EVENT_COLUMNS),
    ):
        missing = _missing_columns(frame, columns)
        if missing:
            findings.append(
                QualityFinding(
                    "missing_columns", "error", len(missing), f"{name}: {sorted(missing)}"
                )
            )
    if findings:
        return findings
    checks = [
        ("null_event_user", events["user_id"].isna(), "event user_id is null"),
        ("null_event_item", events["item_id"].isna(), "event item_id is null"),
        (
            "invalid_event_type",
            ~events["event_type"].isin(EVENT_WEIGHTS),
            "event type is unsupported",
        ),
        (
            "invalid_position",
            events["position"].fillna(0).astype(float) < 1,
            "position must be positive",
        ),
        ("unknown_event_user", ~events["user_id"].isin(users["user_id"]), "event user is unknown"),
        ("unknown_event_item", ~events["item_id"].isin(items["item_id"]), "event item is unknown"),
    ]
    parsed_time = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    checks.append(("invalid_timestamp", parsed_time.isna(), "timestamp is invalid"))
    for code, mask, message in checks:
        count = int(mask.sum())
        if count:
            findings.append(QualityFinding(code, "error", count, message))
    duplicate_count = int(events.duplicated("event_id", keep=False).sum())
    if duplicate_count:
        findings.append(
            QualityFinding("duplicate_event_id", "warning", duplicate_count, "event IDs repeat")
        )
    for name, frame in (("users", users), ("items", items)):
        duplicate_entities = int(frame.duplicated(f"{name[:-1]}_id", keep=False).sum())
        if duplicate_entities:
            findings.append(
                QualityFinding("duplicate_entity", "error", duplicate_entities, f"duplicate {name}")
            )
    return findings


def validate_raw(
    directory: Path, report_path: Path, fail_on_error: bool = False
) -> list[QualityFinding]:
    try:
        users = pd.read_parquet(directory / "users.parquet")
        items = pd.read_parquet(directory / "items.parquet")
        events = pd.read_parquet(directory / "events.parquet")
    except (OSError, ValueError) as error:
        raise DataQualityError(f"cannot read raw Parquet data from {directory}") from error
    findings = validate_frames(users, items, events)
    atomic_write_json(
        report_path,
        {
            "status": "failed" if any(f.severity == "error" for f in findings) else "passed",
            "findings": [asdict(finding) for finding in findings],
            "row_counts": {"users": len(users), "items": len(items), "events": len(events)},
        },
    )
    if fail_on_error and any(f.severity == "error" for f in findings):
        raise DataQualityError(f"raw data failed quality checks; see {report_path}")
    return findings
