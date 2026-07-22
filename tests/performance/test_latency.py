import time

import numpy as np
import pytest

from recommender.indexing.exact import ExactIndex


@pytest.mark.performance
def test_bounded_exact_search_latency() -> None:
    rng = np.random.default_rng(8)
    vectors = rng.normal(size=(2000, 32)).astype(np.float32)
    queries = rng.normal(size=(32, 32)).astype(np.float32)
    index = ExactIndex(vectors, np.asarray([f"i{index}" for index in range(len(vectors))]))
    started = time.perf_counter()
    scores, items = index.search(queries, 20)
    elapsed = time.perf_counter() - started
    assert scores.shape == items.shape == (32, 20)
    assert elapsed < 5.0
