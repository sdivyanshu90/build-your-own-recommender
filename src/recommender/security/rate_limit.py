"""Gateway-compatible rate-limit extension point."""

from typing import Protocol


class RateLimiter(Protocol):
    def allow(self, principal: str, cost: int = 1) -> bool: ...


class AllowAllRateLimiter:
    """Local default; production must inject a distributed limiter at the gateway."""

    def allow(self, principal: str, cost: int = 1) -> bool:
        return bool(principal) and cost > 0
