"""Optional parameterized PostgreSQL repository using an injected connection pool."""

from typing import Any


class PostgresRepository:
    def __init__(self, pool: Any) -> None:
        self.pool = pool

    def seen_items(self, user_id: str) -> set[str]:
        with self.pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT item_id FROM user_seen_items WHERE user_id = %s", (user_id,))
            return {str(row[0]) for row in cursor.fetchall()}

    def is_available(self, item_id: str) -> bool:
        with self.pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT available FROM items WHERE item_id = %s", (item_id,))
            row = cursor.fetchone()
            return bool(row and row[0])

    def delete_user(self, user_id: str) -> int:
        with self.pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM user_seen_items WHERE user_id = %s", (user_id,))
            deleted = int(cursor.rowcount)
            connection.commit()
            return deleted
