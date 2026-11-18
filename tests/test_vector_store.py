import pytest

from src.rag.vector_store import SQLiteVectorStore


@pytest.fixture
def vector_store(tmp_path):
    """Create an isolated temporary vector database."""
    db_path = tmp_path / "test_vectors.db"
    return SQLiteVectorStore(db_path=str(db_path))


@pytest.fixture
def sample_chunk():
    """Provide a valid chunk for testing."""
    return {
        "chunk_id": "chunk-001",
        "document_id": "returns-policy",
        "text": "Customers can request returns within 30 days.",
        "embedding": [0.1, 0.2, 0.3],
        "source_path": "knowledge_base/policies/returns.md",
        "section_title": "Return Eligibility",
        "ordinal": 0,
    }


def test_add_empty_chunks_returns_zero(vector_store):
    result = vector_store.add_chunks([])

    assert result == 0


def test_add_valid_chunk_returns_one(vector_store, sample_chunk):
    result = vector_store.add_chunks([sample_chunk])

    assert result == 1


def test_add_chunks_updates_existing_chunk(
    vector_store,
    sample_chunk,
):
    vector_store.add_chunks([sample_chunk])

    updated_chunk = sample_chunk.copy()
    updated_chunk["text"] = "Updated return policy."

    result = vector_store.add_chunks([updated_chunk])

    assert result == 1

    with vector_store._get_connection() as connection:
        row = connection.execute(
            "SELECT text FROM chunks WHERE chunk_id = ?",
            (sample_chunk["chunk_id"],),
        ).fetchone()

    assert row["text"] == "Updated return policy."


def test_add_chunks_rejects_missing_fields(
    vector_store,
    sample_chunk,
):
    invalid_chunk = sample_chunk.copy()
    del invalid_chunk["document_id"]

    with pytest.raises(ValueError, match="Missing required fields"):
        vector_store.add_chunks([invalid_chunk])


def test_add_chunks_rejects_empty_embedding(
    vector_store,
    sample_chunk,
):
    invalid_chunk = sample_chunk.copy()
    invalid_chunk["embedding"] = []

    with pytest.raises(ValueError, match="non-empty"):
        vector_store.add_chunks([invalid_chunk])


def test_add_chunks_rejects_non_numeric_embedding(
    vector_store,
    sample_chunk,
):
    invalid_chunk = sample_chunk.copy()
    invalid_chunk["embedding"] = [0.1, "invalid", 0.3]

    with pytest.raises(ValueError, match="finite numeric"):
        vector_store.add_chunks([invalid_chunk])


def test_add_chunks_rejects_boolean_embedding(
    vector_store,
    sample_chunk,
):
    invalid_chunk = sample_chunk.copy()
    invalid_chunk["embedding"] = [0.1, True, 0.3]

    with pytest.raises(ValueError, match="finite numeric"):
        vector_store.add_chunks([invalid_chunk])


def test_chunks_persist_after_reopening_database(
    tmp_path,
    sample_chunk,
):
    db_path = tmp_path / "persistent_vectors.db"

    first_store = SQLiteVectorStore(db_path=str(db_path))
    first_store.add_chunks([sample_chunk])

    second_store = SQLiteVectorStore(db_path=str(db_path))

    with second_store._get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?",
            (sample_chunk["chunk_id"],),
        ).fetchone()

    assert row is not None
    assert row["text"] == sample_chunk["text"]




def test_search_returns_most_similar_chunks(
    vector_store,
    sample_chunk,
):
    chunk_a = sample_chunk.copy()
    chunk_a["chunk_id"] = "chunk-a"
    chunk_a["embedding"] = [1.0, 0.0]
    chunk_a["text"] = "Exact match"

    chunk_b = sample_chunk.copy()
    chunk_b["chunk_id"] = "chunk-b"
    chunk_b["embedding"] = [0.8, 0.6]
    chunk_b["text"] = "Close match"
    chunk_b["ordinal"] = 1

    chunk_c = sample_chunk.copy()
    chunk_c["chunk_id"] = "chunk-c"
    chunk_c["embedding"] = [0.0, 1.0]
    chunk_c["text"] = "Unrelated match"
    chunk_c["ordinal"] = 2

    vector_store.add_chunks([chunk_a, chunk_b, chunk_c])

    results = vector_store.search(
        query_embedding=[1.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-a"
    assert results[1]["chunk_id"] == "chunk-b"
    assert results[0]["score"] > results[1]["score"]


def test_search_returns_metadata(vector_store, sample_chunk):
    vector_store.add_chunks([sample_chunk])

    results = vector_store.search([0.1, 0.2, 0.3])

    assert results[0]["document_id"] == "returns-policy"
    assert results[0]["source_path"] == (
        "knowledge_base/policies/returns.md"
    )
    assert results[0]["section_title"] == "Return Eligibility"
    assert "embedding" not in results[0]


def test_search_empty_database_returns_empty_list(vector_store):
    results = vector_store.search([1.0, 0.0])

    assert results == []


def test_search_rejects_invalid_top_k(
    vector_store,
):
    with pytest.raises(ValueError, match="positive integer"):
        vector_store.search([1.0, 0.0], top_k=0)


def test_search_rejects_boolean_top_k(
    vector_store,
):
    with pytest.raises(ValueError, match="positive integer"):
        vector_store.search([1.0, 0.0], top_k=True)


def test_search_rejects_dimension_mismatch(
    vector_store,
    sample_chunk,
):
    vector_store.add_chunks([sample_chunk])

    with pytest.raises(ValueError, match="same dimensions"):
        vector_store.search([1.0, 0.0])


import sqlite3


def make_replacement_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    ordinal: int = 0,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "embedding": [1.0, 0.0, 0.0],
        "source_path": f"{document_id}.md",
        "section_title": "Test Section",
        "ordinal": ordinal,
    }


def test_replace_document_chunks_removes_stale_chunks(tmp_path):
    store = SQLiteVectorStore(
        db_path=str(tmp_path / "replacement.db")
    )

    store.add_chunks([
        make_replacement_chunk(
            "old-1", "returns.md", "Old returns policy"
        ),
        make_replacement_chunk(
            "old-2", "returns.md", "Old refund policy", 1
        ),
    ])

    count = store.replace_document_chunks(
        "returns.md",
        [
            make_replacement_chunk(
                "new-1", "returns.md", "Updated returns policy"
            )
        ],
    )

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert count == 1
    assert len(results) == 1
    assert results[0]["chunk_id"] == "new-1"
    assert results[0]["text"] == "Updated returns policy"


def test_replace_document_chunks_preserves_other_documents(tmp_path):
    store = SQLiteVectorStore(
        db_path=str(tmp_path / "isolation.db")
    )

    store.add_chunks([
        make_replacement_chunk(
            "return-1", "returns.md", "Returns policy"
        ),
        make_replacement_chunk(
            "shipping-1", "shipping.md", "Shipping policy", 1
        ),
    ])

    store.replace_document_chunks(
        "returns.md",
        [
            make_replacement_chunk(
                "return-2", "returns.md", "New returns policy"
            )
        ],
    )

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    chunk_ids = {item["chunk_id"] for item in results}

    assert chunk_ids == {"return-2", "shipping-1"}


def test_replace_document_chunks_rolls_back_on_insert_failure(
    tmp_path,
):
    db_path = tmp_path / "rollback.db"

    store = SQLiteVectorStore(db_path=str(db_path))

    store.add_chunks([
        make_replacement_chunk(
            "old-1", "returns.md", "Original policy"
        )
    ])

    connection = sqlite3.connect(db_path)

    try:
        connection.execute(
            """
            CREATE TRIGGER reject_test_chunk
            BEFORE INSERT ON chunks
            WHEN NEW.text = 'FORCE FAILURE'
            BEGIN
                SELECT RAISE(ABORT, 'Forced test failure');
            END
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(sqlite3.IntegrityError):
        store.replace_document_chunks(
            "returns.md",
            [
                make_replacement_chunk(
                    "new-1",
                    "returns.md",
                    "FORCE FAILURE",
                )
            ],
        )

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "old-1"
    assert results[0]["text"] == "Original policy"


def test_replace_document_chunks_with_empty_list_deletes_document(
    tmp_path,
):
    store = SQLiteVectorStore(
        db_path=str(tmp_path / "empty-replacement.db")
    )

    store.add_chunks([
        make_replacement_chunk(
            "return-1", "returns.md", "Returns policy"
        ),
        make_replacement_chunk(
            "shipping-1", "shipping.md", "Shipping policy", 1
        ),
    ])

    count = store.replace_document_chunks(
        "returns.md",
        [],
    )

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert count == 0
    assert len(results) == 1
    assert results[0]["document_id"] == "shipping.md"



def test_sync_managed_documents_removes_deleted_documents(tmp_path):
    store = SQLiteVectorStore(
        db_path=str(tmp_path / "sync.db")
    )

    returns = make_replacement_chunk(
        "returns-1", "returns.md", "Returns policy"
    )
    shipping = make_replacement_chunk(
        "shipping-1", "shipping.md", "Shipping policy"
    )

    result = store.sync_managed_documents({
        "returns.md": [returns],
        "shipping.md": [shipping],
    })

    assert result["chunks_stored"] == 2
    assert result["documents_removed"] == 0

    # The next inventory no longer contains shipping.md.
    result = store.sync_managed_documents({
        "returns.md": [
            make_replacement_chunk(
                "returns-2", "returns.md", "Updated returns policy"
            )
        ],
    })

    assert result["documents_removed"] == 1

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0]["document_id"] == "returns.md"
    assert results[0]["text"] == "Updated returns policy"


def test_sync_managed_documents_preserves_unmanaged_records(tmp_path):
    store = SQLiteVectorStore(
        db_path=str(tmp_path / "ownership.db")
    )

    store.add_chunks([
        make_replacement_chunk(
            "legacy-1", "legacy.md", "Legacy test record"
        )
    ])

    store.sync_managed_documents({
        "returns.md": [
            make_replacement_chunk(
                "returns-1", "returns.md", "Returns policy"
            )
        ],
    })

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert {item["chunk_id"] for item in results} == {
        "legacy-1",
        "returns-1",
    }


def test_sync_managed_documents_rejects_empty_inventory_by_default(
    tmp_path,
):
    store = SQLiteVectorStore(
        db_path=str(tmp_path / "empty.db")
    )

    with pytest.raises(ValueError, match="empty document inventory"):
        store.sync_managed_documents({})


def test_sync_managed_documents_requires_explicit_empty_permission(
    tmp_path,
):
    store = SQLiteVectorStore(
        db_path=str(tmp_path / "clear.db")
    )

    store.sync_managed_documents({
        "returns.md": [
            make_replacement_chunk(
                "returns-1", "returns.md", "Returns policy"
            )
        ],
    })

    result = store.sync_managed_documents(
        {},
        allow_empty=True,
    )

    assert result["documents_removed"] == 1
    assert result["chunks_stored"] == 0
    assert store.search([1.0, 0.0, 0.0]) == []


def test_sync_managed_documents_rejects_unmanaged_chunk_id_collision(
    tmp_path,
):
    store = SQLiteVectorStore(
        db_path=str(tmp_path / "collision.db")
    )

    store.add_chunks([
        make_replacement_chunk(
            "shared-id", "manual.md", "Manual record"
        )
    ])

    with pytest.raises(ValueError, match="unmanaged"):
        store.sync_managed_documents({
            "returns.md": [
                make_replacement_chunk(
                    "shared-id", "returns.md", "Returns policy"
                )
            ],
        })

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0]["document_id"] == "manual.md"
    assert results[0]["text"] == "Manual record"


def test_sync_managed_documents_rolls_back_on_database_failure(
    tmp_path,
):
    import sqlite3

    db_path = tmp_path / "sync-rollback.db"
    store = SQLiteVectorStore(db_path=str(db_path))

    store.sync_managed_documents({
        "returns.md": [
            make_replacement_chunk(
                "old-1", "returns.md", "Original policy"
            )
        ],
    })

    connection = sqlite3.connect(db_path)

    try:
        connection.execute(
            """
            CREATE TRIGGER reject_sync_chunk
            BEFORE INSERT ON chunks
            WHEN NEW.text = 'FORCE FAILURE'
            BEGIN
                SELECT RAISE(ABORT, 'Forced sync failure');
            END
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(sqlite3.IntegrityError):
        store.sync_managed_documents({
            "returns.md": [
                make_replacement_chunk(
                    "new-1",
                    "returns.md",
                    "FORCE FAILURE",
                )
            ],
        })

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "old-1"
    assert results[0]["text"] == "Original policy"
