import pytest
from hypothesis import given
from hypothesis import strategies as st

from recommender.evaluation.metrics import (
    catalog_coverage,
    category_diversity,
    novelty,
    ranking_metrics,
)


def test_golden_ranking_metrics() -> None:
    result = ranking_metrics(["a", "x", "b"], {"a", "b"}, 3)
    assert result["recall"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["mrr"] == 1
    assert result["map"] == pytest.approx((1 + 2 / 3) / 2)


@given(
    st.lists(st.text(min_size=1), unique=True, max_size=30), st.integers(min_value=1, max_value=30)
)
def test_metric_bounds(recommended, k) -> None:
    relevant = set(recommended[::2])
    for value in ranking_metrics(recommended, relevant, k).values():
        assert 0 <= value <= 1


def test_coverage_and_diversity() -> None:
    rows = [["a", "b"], ["b", "c"]]
    assert catalog_coverage(rows, {"a", "b", "c", "d"}) == 0.75
    assert category_diversity(rows, {"a": "x", "b": "y", "c": "y"}) == 0.5


def test_metric_edge_cases_are_explicit() -> None:
    with pytest.raises(ValueError, match="positive"):
        ranking_metrics([], set(), 0)
    assert catalog_coverage([[]], set()) == 0
    assert category_diversity([["only"]], {"only": "x"}) == 0
    assert novelty([], {}) == 0
