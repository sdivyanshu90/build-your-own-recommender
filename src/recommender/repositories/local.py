"""In-memory repository used by tests and local serving."""

from collections.abc import Mapping


class LocalRepository:
    def __init__(self, histories: Mapping[str, set[str]], availability: Mapping[str, bool]) -> None:
        self.histories = {key: set(value) for key, value in histories.items()}
        self.availability = dict(availability)

    def seen_items(self, user_id: str) -> set[str]:
        return set(self.histories.get(user_id, set()))

    def is_available(self, item_id: str) -> bool:
        return self.availability.get(item_id, False)

    def delete_user(self, user_id: str) -> int:
        return int(self.histories.pop(user_id, None) is not None)
