"""Thread-safe bounded in-memory TTL cache."""

import threading
import time
from collections import OrderedDict


class InMemoryCache[T]:
    def __init__(self, max_entries: int = 10000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._values: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> T | None:
        with self._lock:
            record = self._values.get(key)
            if record is None:
                return None
            expires, value = record
            if expires <= time.monotonic():
                del self._values[key]
                return None
            self._values.move_to_end(key)
            return value

    def set(self, key: str, value: T, ttl_seconds: float) -> None:
        with self._lock:
            self._values[key] = (time.monotonic() + ttl_seconds, value)
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
