"""Filtering and lightweight relevance/diversity trade-off policies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    item_id: str
    retrieval_score: float
    category: str
    popularity: float
    freshness_days: float
    available: bool
    final_score: float | None = None


@dataclass(frozen=True)
class RerankConfig:
    freshness_weight: float = 0.0
    diversity_weight: float = 0.15
    popularity_cap_per_category: int | None = None


def apply_policies(
    candidates: list[Candidate], limit: int, config: RerankConfig
) -> list[Candidate]:
    """Greedy category-aware MMR after availability and deduplication."""
    unique: dict[str, Candidate] = {}
    for candidate in candidates:
        if candidate.available and candidate.item_id not in unique:
            freshness = 1 / (1 + max(candidate.freshness_days, 0))
            score = candidate.retrieval_score + config.freshness_weight * freshness
            unique[candidate.item_id] = Candidate(**{**candidate.__dict__, "final_score": score})
    remaining = list(unique.values())
    selected: list[Candidate] = []
    category_counts: dict[str, int] = {}
    while remaining and len(selected) < limit:
        eligible = [
            candidate
            for candidate in remaining
            if config.popularity_cap_per_category is None
            or category_counts.get(candidate.category, 0) < config.popularity_cap_per_category
        ]
        if not eligible:
            break
        choice = max(
            eligible,
            key=lambda candidate: (
                (candidate.final_score or candidate.retrieval_score)
                - config.diversity_weight * category_counts.get(candidate.category, 0)
            ),
        )
        selected.append(choice)
        category_counts[choice.category] = category_counts.get(choice.category, 0) + 1
        remaining.remove(choice)
    return selected
