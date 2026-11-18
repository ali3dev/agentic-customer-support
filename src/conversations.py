import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConversationNotFound(LookupError):
    """Raised when a conversation is outside the caller's ownership scope."""


class ConversationStore:
    """Persist conversations with tenant and optional customer ownership."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        connection = self._connect()

        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        customer_id TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(conversations)"
                    ).fetchall()
                }
                if "customer_id" not in columns:
                    connection.execute(
                        """
                        ALTER TABLE conversations
                        ADD COLUMN customer_id TEXT
                        """
                    )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversations_tenant
                    ON conversations(tenant_id, updated_at)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_conversations_tenant_customer_updated
                    ON conversations(tenant_id, customer_id, updated_at)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        conversation_id TEXT NOT NULL
                            REFERENCES conversations(conversation_id)
                            ON DELETE CASCADE,
                        message_index INTEGER NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        sources_json TEXT NOT NULL,
                        usage_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (conversation_id, message_index)
                    )
                    """
                )
        finally:
            connection.close()

    def save_turn(
        self,
        tenant_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        sources: list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
        customer_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        connection = self._connect()

        try:
            connection.execute("BEGIN IMMEDIATE")
            conversation = connection.execute(
                """
                SELECT tenant_id, customer_id
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

            if conversation is not None:
                if conversation["tenant_id"] != tenant_id:
                    raise ConversationNotFound(conversation_id)
                if (
                    customer_id is not None
                    and conversation["customer_id"] != customer_id
                ):
                    raise ConversationNotFound(conversation_id)

            if conversation is None:
                connection.execute(
                    """
                    INSERT INTO conversations (
                        conversation_id, tenant_id, customer_id,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (conversation_id, tenant_id, customer_id, now, now),
                )

            next_index = connection.execute(
                """
                SELECT COALESCE(MAX(message_index), -1) + 1
                FROM conversation_messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()[0]

            connection.executemany(
                """
                INSERT INTO conversation_messages (
                    conversation_id, message_index, role, content,
                    sources_json, usage_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        conversation_id,
                        next_index,
                        "user",
                        user_message,
                        "[]",
                        "{}",
                        now,
                    ),
                    (
                        conversation_id,
                        next_index + 1,
                        "assistant",
                        assistant_message,
                        json.dumps(sources),
                        json.dumps(usage or {}),
                        now,
                    ),
                ],
            )
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE conversation_id = ? AND tenant_id = ?
                """,
                (now, conversation_id, tenant_id),
            )
            connection.commit()
        finally:
            connection.close()

    def get_history(
        self,
        tenant_id: str,
        conversation_id: str,
        limit: int = 20,
        customer_id: str | None = None,
    ) -> list[dict[str, str]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

        connection = self._connect()

        try:
            query = """
                SELECT conversation_id
                FROM conversations
                WHERE conversation_id = ? AND tenant_id = ?
            """
            params: tuple[str, ...] = (conversation_id, tenant_id)
            if customer_id is not None:
                query += " AND customer_id = ?"
                params += (customer_id,)
            conversation = connection.execute(query, params).fetchone()

            if conversation is None:
                raise ConversationNotFound(conversation_id)

            rows = connection.execute(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY message_index DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        finally:
            connection.close()

        return [
            {
                "role": row["role"],
                "content": row["content"],
            }
            for row in reversed(rows)
        ]

    def get_messages(
        self,
        tenant_id: str,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            query = """
                SELECT conversation_id
                FROM conversations
                WHERE conversation_id = ? AND tenant_id = ?
            """
            params: tuple[str, ...] = (conversation_id, tenant_id)
            if customer_id is not None:
                query += " AND customer_id = ?"
                params += (customer_id,)
            conversation = connection.execute(query, params).fetchone()

            if conversation is None:
                raise ConversationNotFound(conversation_id)

            rows = connection.execute(
                """
                SELECT role, content, sources_json, usage_json, created_at
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY message_index
                """,
                (conversation_id,),
            ).fetchall()
        finally:
            connection.close()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources_json"]),
                "usage": json.loads(row["usage_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def list_conversations(
        self,
        tenant_id: str,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            query = """
                SELECT
                    conversations.conversation_id,
                    conversations.created_at,
                    conversations.updated_at,
                    COUNT(conversation_messages.message_index) AS message_count
                FROM conversations
                LEFT JOIN conversation_messages
                    ON conversation_messages.conversation_id =
                       conversations.conversation_id
                WHERE conversations.tenant_id = ?
            """
            params: tuple[str, ...] = (tenant_id,)
            if customer_id is not None:
                query += " AND conversations.customer_id = ?"
                params += (customer_id,)
            query += """
                GROUP BY conversations.conversation_id
                ORDER BY conversations.updated_at DESC
            """
            rows = connection.execute(query, params).fetchall()
        finally:
            connection.close()

        return [dict(row) for row in rows]

    def usage_summary(self, tenant_id: str) -> dict[str, Any]:
        connection = self._connect()

        try:
            rows = connection.execute(
                """
                SELECT conversation_messages.usage_json
                FROM conversation_messages
                JOIN conversations
                    ON conversations.conversation_id =
                       conversation_messages.conversation_id
                WHERE conversations.tenant_id = ?
                  AND conversation_messages.role = 'assistant'
                """,
                (tenant_id,),
            ).fetchall()
        finally:
            connection.close()

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost_usd = 0.0
        cost_records = 0

        for row in rows:
            usage = json.loads(row["usage_json"])
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
            if usage.get("cost_usd") is not None:
                cost_usd += usage["cost_usd"]
                cost_records += 1

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd if cost_records else None,
            "cost_records": cost_records,
        }

    def delete_conversation(
        self,
        tenant_id: str,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> bool:
        connection = self._connect()

        try:
            with connection:
                query = """
                    DELETE FROM conversations
                    WHERE conversation_id = ? AND tenant_id = ?
                """
                params: tuple[str, ...] = (conversation_id, tenant_id)
                if customer_id is not None:
                    query += " AND customer_id = ?"
                    params += (customer_id,)
                cursor = connection.execute(query, params)
        finally:
            connection.close()

        return cursor.rowcount == 1
