from hypothesis import given
from hypothesis import strategies as st

from recommender.reranking.policies import Candidate, RerankConfig, apply_policies


@given(
    st.lists(st.integers(min_value=0, max_value=20), max_size=50),
    st.integers(min_value=1, max_value=20),
)
def test_reranking_is_bounded_deduplicated_and_available(item_numbers, limit) -> None:
    candidates = [
        Candidate(str(number), float(number), str(number % 3), 0.1, 1, number % 5 != 0)
        for number in item_numbers
    ]
    result = apply_policies(candidates, limit, RerankConfig())
    assert len(result) <= limit
    assert len({item.item_id for item in result}) == len(result)
    assert all(item.available for item in result)
