import pandas as pd
import pytest

from recommender.caching.redis import RedisCache
from recommender.monitoring.drift import compare_frames
from recommender.repositories.local import LocalRepository
from recommender.security.identifiers import hash_identifier
from recommender.security.rate_limit import AllowAllRateLimiter


class FakeRedis:
    def __init__(self) -> None:
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        assert ttl >= 1
        self.values[key] = value


def test_redis_adapter_round_trip_and_miss() -> None:
    cache = RedisCache(FakeRedis(), "test:")
    assert cache.get("missing") is None
    cache.set("key", {"safe": True}, 0.1)
    assert cache.get("key") == {"safe": True}
    cache.clear()


def test_drift_report_covers_numeric_categorical_and_missingness() -> None:
    reference = pd.DataFrame({"numeric": range(20), "category": ["a"] * 20})
    current = pd.DataFrame({"numeric": range(10, 30), "category": ["a", "b"] * 10})
    report = compare_frames(reference, current)
    assert report["row_count_ratio"] == 1
    assert report["columns"]["numeric"]["psi"] >= 0
    assert report["columns"]["category"]["unknown_category_rate"] == 0.5


def test_local_repository_deletion_and_availability() -> None:
    repository = LocalRepository({"u": {"a"}}, {"a": True})
    assert repository.seen_items("u") == {"a"}
    assert repository.seen_items("missing") == set()
    assert repository.is_available("a")
    assert not repository.is_available("missing")
    assert repository.delete_user("u") == 1
    assert repository.delete_user("u") == 0


def test_identifier_hash_is_stable_pseudonymous_and_requires_salt() -> None:
    first = hash_identifier("user", "a-long-deployment-salt")
    assert first == hash_identifier("user", "a-long-deployment-salt")
    assert first != hash_identifier("other", "a-long-deployment-salt")
    assert "user" not in first
    with pytest.raises(ValueError, match="salt"):
        hash_identifier("user", "short")


def test_local_rate_limit_extension() -> None:
    limiter = AllowAllRateLimiter()
    assert limiter.allow("principal")
    assert not limiter.allow("")
    assert not limiter.allow("principal", 0)
