"""Cache boundary."""

from typing import Protocol, TypeVar

T = TypeVar("T")


class Cache(Protocol[T]):
    def get(self, key: str) -> T | None: ...

    def set(self, key: str, value: T, ttl_seconds: float) -> None: ...

    def clear(self) -> None: ...
