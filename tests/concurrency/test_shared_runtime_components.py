from concurrent.futures import ThreadPoolExecutor

import numpy as np

from recommender.caching.memory import InMemoryCache
from recommender.indexing.exact import ExactIndex


def test_shared_index_and_cache_are_safe_under_concurrent_reads() -> None:
    vectors = np.eye(32, dtype=np.float32)
    index = ExactIndex(vectors, np.asarray([str(value) for value in range(32)]))
    cache = InMemoryCache[list[str]](max_entries=64)

    def retrieve(position: int) -> list[str]:
        key = str(position)
        cached = cache.get(key)
        if cached is not None:
            return cached
        _, items = index.search(vectors[position : position + 1], 3)
        result = items[0].tolist()
        cache.set(key, result, 30)
        return result

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(retrieve, [value % 32 for value in range(256)]))
    assert len(results) == 256
    assert all(result for result in results)
