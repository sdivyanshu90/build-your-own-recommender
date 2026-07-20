import numpy as np
import pytest

from recommender.exceptions import ConfigurationError
from recommender.sampling.samplers import HardNegativeSampler, PopularitySampler, UniformSampler


def test_uniform_sampler_is_deterministic_and_excludes_positives() -> None:
    first = UniformSampler([1, 2, 3, 4], 4).sample(1, 20, {2})
    second = UniformSampler([1, 2, 3, 4], 4).sample(1, 20, {2})
    np.testing.assert_array_equal(first, second)
    assert set(first) <= {3, 4}


def test_popularity_sampler_favors_popular_items() -> None:
    sample = PopularitySampler([1, 2, 3], [1, 10, 100], 2, alpha=1).sample(1, 2000)
    assert np.sum(sample == 3) > np.sum(sample == 2) * 5


def test_hard_sampler_excludes_known_positive() -> None:
    def search(query, k):
        del query, k
        return np.ones((1, 4)), np.array([[1, 2, 3, 4]])

    sample = HardNegativeSampler(search).sample(np.ones(2), 1, 2, {2})
    np.testing.assert_array_equal(sample, [3, 4])


def test_invalid_popularity_fails() -> None:
    with pytest.raises(ConfigurationError):
        PopularitySampler([1, 2], [0, 0], 1)


def test_samplers_fail_when_no_eligible_candidates() -> None:
    with pytest.raises(ConfigurationError, match="at least one"):
        UniformSampler([], 1)
    with pytest.raises(ConfigurationError, match="eligible"):
        UniformSampler([1], 1).sample(1, 1)
    with pytest.raises(ConfigurationError, match="eligible"):
        PopularitySampler([1], [1], 1).sample(1, 1)

    def short_search(query, k):
        del query, k
        return np.ones((1, 1)), np.array([[1]])

    with pytest.raises(ConfigurationError, match="too few"):
        HardNegativeSampler(short_search).sample(np.ones(2), 1, 1)
