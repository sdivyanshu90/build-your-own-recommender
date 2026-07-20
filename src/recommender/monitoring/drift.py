"""Offline feature and missingness drift utilities."""

from typing import Any

import numpy as np
import pandas as pd


def _population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(reference_values) or not len(current_values):
        return 0.0
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_hist = np.histogram(reference_values, edges)[0] / len(reference_values)
    current_hist = np.histogram(current_values, edges)[0] / len(current_values)
    reference_hist = np.clip(reference_hist, 1e-6, None)
    current_hist = np.clip(current_hist, 1e-6, None)
    return float(np.sum((current_hist - reference_hist) * np.log(current_hist / reference_hist)))


def compare_frames(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    common = sorted(set(reference.columns) & set(current.columns))
    report: dict[str, Any] = {
        "row_count_ratio": len(current) / max(len(reference), 1),
        "columns": {},
    }
    for column in common:
        detail: dict[str, float] = {
            "reference_missing_rate": float(reference[column].isna().mean()),
            "current_missing_rate": float(current[column].isna().mean()),
        }
        if pd.api.types.is_numeric_dtype(reference[column]):
            detail["psi"] = _population_stability_index(reference[column], current[column])
        else:
            known = set(reference[column].dropna().astype(str))
            current_values = current[column].dropna().astype(str)
            detail["unknown_category_rate"] = (
                float((~current_values.isin(known)).mean()) if len(current_values) else 0.0
            )
        report["columns"][column] = detail
    return report
