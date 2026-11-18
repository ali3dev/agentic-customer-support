
import pytest

from src.rag.ingestion import ingest_directory
from src.rag.vector_store import SQLiteVectorStore


class FakeEmbeddingService:
    """Generate predictable vectors for ingestion tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]


class IncorrectEmbeddingService:
    """Return fewer embeddings than requested."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []


@pytest.fixture
def vector_store(tmp_path):
    db_path = tmp_path / "test_vectors.db"
    return SQLiteVectorStore(db_path=str(db_path))


@pytest.fixture
def embedding_service():
    return FakeEmbeddingService()


def test_ingest_directory_stores_markdown_documents(
    tmp_path,
    vector_store,
    embedding_service,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    (knowledge_base / "returns.md").write_text(
        "# Returns\n\nCustomers can request returns within 30 days.",
        encoding="utf-8",
    )

    (knowledge_base / "shipping.md").write_text(
        "# Shipping\n\nStandard shipping takes 3 to 5 days.",
        encoding="utf-8",
    )

    result = ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    assert result["documents_found"] == 2
    assert result["documents_ingested"] == 2
    assert result["chunks_created"] == 2
    assert result["chunks_stored"] == 2

    retrieved = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert len(retrieved) == 2


def test_ingest_directory_skips_empty_documents(
    tmp_path,
    vector_store,
    embedding_service,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    (knowledge_base / "empty.md").write_text(
        "   \n  ",
        encoding="utf-8",
    )

    result = ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    assert result["documents_found"] == 1
    assert result["documents_ingested"] == 0
    assert result["chunks_created"] == 0
    assert result["chunks_stored"] == 0


def test_ingest_directory_is_idempotent(
    tmp_path,
    vector_store,
    embedding_service,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    (knowledge_base / "returns.md").write_text(
        "# Returns\n\nReturns are allowed within 30 days.",
        encoding="utf-8",
    )

    first_result = ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    second_result = ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    assert first_result["chunks_created"] == 1
    assert second_result["chunks_created"] == 1

    retrieved = vector_store.search([1.0, 0.0, 0.0])

    assert len(retrieved) == 1


def test_ingest_directory_rejects_missing_directory(
    tmp_path,
    vector_store,
    embedding_service,
):
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        ingest_directory(
            missing_path,
            embedding_service,
            vector_store,
        )


def test_ingest_directory_rejects_file_as_directory(
    tmp_path,
    vector_store,
    embedding_service,
):
    file_path = tmp_path / "not_a_directory.md"
    file_path.write_text("# Example", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        ingest_directory(
            file_path,
            embedding_service,
            vector_store,
        )


def test_ingest_directory_rejects_embedding_count_mismatch(
    tmp_path,
    vector_store,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    (knowledge_base / "returns.md").write_text(
        "# Returns\n\nReturns are allowed within 30 days.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Embedding count does not match chunk count",
    ):
        ingest_directory(
            knowledge_base,
            IncorrectEmbeddingService(),
            vector_store,
        )


def test_reingestion_removes_stale_chunks_after_document_changes(
    tmp_path,
    vector_store,
    embedding_service,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    document = knowledge_base / "returns.md"

    document.write_text(
        "# Returns\n\nOld policy allows returns within 30 days.",
        encoding="utf-8",
    )

    ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    old_ids = {
        item["chunk_id"]
        for item in vector_store.search(
            [1.0, 0.0, 0.0],
            top_k=100,
        )
    }

    document.write_text(
        "# Returns\n\nNew policy allows returns within 60 days.",
        encoding="utf-8",
    )

    ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    results = vector_store.search(
        [1.0, 0.0, 0.0],
        top_k=100,
    )

    new_ids = {item["chunk_id"] for item in results}

    assert old_ids.isdisjoint(new_ids)
    assert len(new_ids) == 1
    assert "60 days" in results[0]["text"]


def test_reingestion_of_empty_document_removes_old_chunks(
    tmp_path,
    vector_store,
    embedding_service,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    document = knowledge_base / "returns.md"

    document.write_text(
        "# Returns\n\nReturns are allowed within 30 days.",
        encoding="utf-8",
    )

    ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    document.write_text("   \n ", encoding="utf-8")

    result = ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    assert result["documents_found"] == 1
    assert result["documents_ingested"] == 0

    results = vector_store.search(
        [1.0, 0.0, 0.0],
        top_k=100,
    )

    assert results == []


def test_failed_reingestion_preserves_previous_document_chunks(
    tmp_path,
    vector_store,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    document = knowledge_base / "returns.md"

    document.write_text(
        "# Returns\n\nOriginal returns policy.",
        encoding="utf-8",
    )

    ingest_directory(
        knowledge_base,
        FakeEmbeddingService(),
        vector_store,
    )

    original_results = vector_store.search(
        [1.0, 0.0, 0.0],
        top_k=100,
    )

    original_ids = {
        item["chunk_id"]
        for item in original_results
    }

    document.write_text(
        "# Returns\n\nUpdated returns policy.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Embedding count does not match chunk count",
    ):
        ingest_directory(
            knowledge_base,
            IncorrectEmbeddingService(),
            vector_store,
        )

    current_results = vector_store.search(
        [1.0, 0.0, 0.0],
        top_k=100,
    )

    current_ids = {
        item["chunk_id"]
        for item in current_results
    }

    assert current_ids == original_ids
    assert "Original returns policy." in current_results[0]["text"]



def test_ingestion_removes_deleted_documents(
    tmp_path,
    vector_store,
    embedding_service,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    returns = knowledge_base / "returns.md"
    shipping = knowledge_base / "shipping.md"

    returns.write_text(
        "# Returns\n\nReturns within 30 days.",
        encoding="utf-8",
    )
    shipping.write_text(
        "# Shipping\n\nShipping takes 3 to 5 days.",
        encoding="utf-8",
    )

    ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    shipping.unlink()

    result = ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    assert result["documents_found"] == 1
    assert result["documents_removed"] == 1

    results = vector_store.search(
        [1.0, 0.0, 0.0],
        top_k=100,
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "returns.md"


def test_embedding_failure_preserves_entire_previous_index(
    tmp_path,
    vector_store,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    returns = knowledge_base / "returns.md"
    shipping = knowledge_base / "shipping.md"

    returns.write_text(
        "# Returns\n\nOriginal returns policy.",
        encoding="utf-8",
    )

    ingest_directory(
        knowledge_base,
        FakeEmbeddingService(),
        vector_store,
    )

    original_results = vector_store.search(
        [1.0, 0.0, 0.0],
        top_k=100,
    )

    original_ids = {
        item["chunk_id"]
        for item in original_results
    }

    returns.write_text(
        "# Returns\n\nUpdated returns policy.",
        encoding="utf-8",
    )

    shipping.write_text(
        "# Shipping\n\nNew shipping policy.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Embedding count does not match chunk count",
    ):
        ingest_directory(
            knowledge_base,
            IncorrectEmbeddingService(),
            vector_store,
        )

    current_results = vector_store.search(
        [1.0, 0.0, 0.0],
        top_k=100,
    )

    current_ids = {
        item["chunk_id"]
        for item in current_results
    }

    assert current_ids == original_ids
    assert "Original returns policy." in current_results[0]["text"]


def test_empty_knowledge_base_requires_explicit_permission(
    tmp_path,
    vector_store,
    embedding_service,
):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()

    document = knowledge_base / "returns.md"
    document.write_text(
        "# Returns\n\nReturns within 30 days.",
        encoding="utf-8",
    )

    ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
    )

    document.unlink()

    with pytest.raises(ValueError, match="no Markdown files"):
        ingest_directory(
            knowledge_base,
            embedding_service,
            vector_store,
        )

    # Explicit permission is required to clear the managed index.
    result = ingest_directory(
        knowledge_base,
        embedding_service,
        vector_store,
        allow_empty=True,
    )

    assert result["documents_found"] == 0
    assert result["documents_removed"] == 1
    assert result["chunks_stored"] == 0
    assert vector_store.search([1.0, 0.0, 0.0]) == []
