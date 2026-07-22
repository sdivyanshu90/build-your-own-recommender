"""Repository protocol for user history and item eligibility."""

from typing import Protocol


class RecommendationRepository(Protocol):
    def seen_items(self, user_id: str) -> set[str]: ...

    def is_available(self, item_id: str) -> bool: ...

    def delete_user(self, user_id: str) -> int: ...
