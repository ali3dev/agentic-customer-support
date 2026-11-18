import sqlite3

import pytest

from src.conversations import ConversationNotFound, ConversationStore


def test_customer_scope_is_enforced_by_conversation_store(tmp_path):
    store = ConversationStore(tmp_path / "conversations.db")
    store.save_turn(
        tenant_id="tenant-a",
        conversation_id="conversation-a",
        user_message="Owner question",
        assistant_message="Owner answer",
        sources=[],
        customer_id="customer-a",
    )
    store.save_turn(
        tenant_id="tenant-a",
        conversation_id="conversation-b",
        user_message="Other customer question",
        assistant_message="Other customer answer",
        sources=[],
        customer_id="customer-b",
    )

    assert [
        item["conversation_id"]
        for item in store.list_conversations("tenant-a", "customer-a")
    ] == ["conversation-a"]
    assert store.get_history(
        "tenant-a",
        "conversation-a",
        customer_id="customer-a",
    ) == [
        {"role": "user", "content": "Owner question"},
        {"role": "assistant", "content": "Owner answer"},
    ]

    with pytest.raises(ConversationNotFound):
        store.get_history(
            "tenant-a",
            "conversation-a",
            customer_id="customer-b",
        )
    with pytest.raises(ConversationNotFound):
        store.get_messages(
            "tenant-a",
            "conversation-a",
            customer_id="customer-b",
        )
    assert not store.delete_conversation(
        "tenant-a",
        "conversation-a",
        customer_id="customer-b",
    )
    with pytest.raises(ConversationNotFound):
        store.save_turn(
            tenant_id="tenant-a",
            conversation_id="conversation-a",
            user_message="Attempted takeover",
            assistant_message="Must not be saved",
            sources=[],
            customer_id="customer-b",
        )

    assert len(
        store.get_messages(
            "tenant-a",
            "conversation-a",
            customer_id="customer-a",
        )
    ) == 2
    assert len(store.list_conversations("tenant-a")) == 2


def test_legacy_conversations_without_customer_owner_are_admin_only(
    tmp_path,
):
    database = tmp_path / "legacy_conversations.db"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO conversations (
            conversation_id, tenant_id, created_at, updated_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            "legacy-conversation",
            "tenant-a",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    store = ConversationStore(database)

    assert [
        item["conversation_id"]
        for item in store.list_conversations("tenant-a")
    ] == ["legacy-conversation"]
    assert store.list_conversations("tenant-a", "customer-a") == []
    with pytest.raises(ConversationNotFound):
        store.get_messages(
            "tenant-a",
            "legacy-conversation",
            customer_id="customer-a",
        )
