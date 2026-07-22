"""Redis-compatible cache adapter with an injected client."""

import json
from typing import Any


class RedisCache:
    def __init__(self, client: Any, prefix: str = "recommender:") -> None:
        self.client = client
        self.prefix = prefix

    def get(self, key: str) -> Any | None:
        value = self.client.get(self.prefix + key)
        if value is None:
            return None
        return json.loads(value)

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        self.client.setex(
            self.prefix + key, max(1, int(ttl_seconds)), json.dumps(value, sort_keys=True)
        )

    def clear(self) -> None:
        """Do not perform a production-wide wildcard delete; version prefixes expire naturally."""
