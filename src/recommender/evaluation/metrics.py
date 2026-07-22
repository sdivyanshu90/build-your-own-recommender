"""Well-defined ranking, coverage, novelty, and diversity metrics."""

import math
from collections.abc import Mapping, Sequence

import numpy as np


def ranking_metrics(recommended: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    """Binary-relevance metrics; precision denominator is K even for short lists."""
    if k < 1:
        raise ValueError("k must be positive")
    ranked = list(recommended[:k])
    hits = np.asarray([item in relevant for item in ranked], dtype=float)
    hit_count = float(hits.sum())
    recall = hit_count / len(relevant) if relevant else 0.0
    precision = hit_count / k
    hit_rate = float(hit_count > 0)
    hit_positions = np.flatnonzero(hits)
    mrr = 1 / float(hit_positions[0] + 1) if len(hit_positions) else 0.0
    discounts = 1 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float((hits * discounts).sum())
    ideal_count = min(len(relevant), k)
    idcg = float((1 / np.log2(np.arange(2, ideal_count + 2))).sum()) if ideal_count else 0.0
    ndcg = dcg / idcg if idcg else 0.0
    precisions = [float(hits[: position + 1].mean()) for position in hit_positions]
    average_precision = sum(precisions) / min(len(relevant), k) if relevant else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "ndcg": ndcg,
        "map": average_precision,
    }


def catalog_coverage(recommendations: Sequence[Sequence[str]], catalog: set[str]) -> float:
    if not catalog:
        return 0.0
    return len({item for row in recommendations for item in row} & catalog) / len(catalog)


def novelty(recommendations: Sequence[Sequence[str]], popularity: Mapping[str, float]) -> float:
    values = [
        -math.log2(max(popularity.get(item, 1e-12), 1e-12))
        for row in recommendations
        for item in row
    ]
    return float(np.mean(values)) if values else 0.0


def category_diversity(
    recommendations: Sequence[Sequence[str]], categories: Mapping[str, str]
) -> float:
    """Mean pairwise category dissimilarity."""
    values: list[float] = []
    for row in recommendations:
        if len(row) < 2:
            continue
        pairs = 0
        different = 0
        for left in range(len(row)):
            for right in range(left + 1, len(row)):
                pairs += 1
                different += categories.get(row[left]) != categories.get(row[right])
        values.append(different / pairs)
    return float(np.mean(values)) if values else 0.0
