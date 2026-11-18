
import pytest

from src.rag.retrieval import RetrievalService


class FakeEmbeddingService:
    """Simulate query embedding without loading a real model."""

    def __init__(self, embedding=None, error=None):
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.error = error
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)

        if self.error:
            raise self.error

        return self.embedding


class FakeVectorStore:
    """Simulate semantic search without using SQLite."""

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.search_calls = []

    def search(self, query_embedding, top_k=5):
        self.search_calls.append({
            "query_embedding": query_embedding,
            "top_k": top_k,
        })

        if self.error:
            raise self.error

        return self.results


@pytest.fixture
def sample_results():
    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "returns.md",
            "text": "Damaged products should be reported to support.",
            "source_path": "policies/returns.md",
            "section_title": "Damaged Products",
            "ordinal": 0,
            "score": 0.91,
        }
    ]


@pytest.fixture
def retrieval_components(sample_results):
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(results=sample_results)

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    return retrieval_service, embedding_service, vector_store


def test_retrieve_returns_relevant_chunks(retrieval_components, sample_results):
    retrieval_service, _, _ = retrieval_components

    results = retrieval_service.retrieve(
        "Can I return a damaged product?"
    )

    assert results == sample_results


def test_retrieve_filters_chunks_below_relevance_threshold(sample_results):
    low_score_result = sample_results[0].copy()
    low_score_result.update({
        "chunk_id": "chunk-low",
        "score": 0.59,
    })
    vector_store = FakeVectorStore(
        results=[sample_results[0], low_score_result]
    )
    retrieval_service = RetrievalService(
        FakeEmbeddingService(),
        vector_store,
    )

    results = retrieval_service.retrieve("Can I return a damaged product?")

    assert [result["chunk_id"] for result in results] == ["chunk-001"]


@pytest.mark.parametrize(
    "threshold",
    [True, "0.6", -1.1, 1.1, float("nan")],
)
def test_retrieval_rejects_invalid_relevance_threshold(threshold):
    with pytest.raises(ValueError, match="between -1 and 1"):
        RetrievalService(
            FakeEmbeddingService(),
            FakeVectorStore(),
            relevance_threshold=threshold,
        )


def test_retrieve_embeds_customer_query(retrieval_components):
    retrieval_service, embedding_service, _ = retrieval_components

    retrieval_service.retrieve("When will my refund arrive?")

    assert embedding_service.queries == [
        "When will my refund arrive?"
    ]


def test_retrieve_passes_embedding_to_vector_store(retrieval_components):
    retrieval_service, embedding_service, vector_store = retrieval_components

    retrieval_service.retrieve("Where is my order?")

    assert vector_store.search_calls[0]["query_embedding"] == (
        embedding_service.embedding
    )


def test_retrieve_forwards_top_k(retrieval_components):
    retrieval_service, _, vector_store = retrieval_components

    retrieval_service.retrieve(
        "What is your return policy?",
        top_k=3,
    )

    assert vector_store.search_calls[0]["top_k"] == 3


def test_retrieve_returns_empty_list_when_no_results():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(results=[])

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = retrieval_service.retrieve("Unknown topic")

    assert results == []


@pytest.mark.parametrize(
    "query",
    ["", "   ", None, 123],
)
def test_retrieve_rejects_invalid_query(retrieval_components, query):
    retrieval_service, embedding_service, vector_store = retrieval_components

    with pytest.raises(
        ValueError,
        match="Query must be a non-empty string",
    ):
        retrieval_service.retrieve(query)

    assert embedding_service.queries == []
    assert vector_store.search_calls == []


def test_retrieve_propagates_embedding_error():
    embedding_service = FakeEmbeddingService(
        error=RuntimeError("Embedding model failed")
    )
    vector_store = FakeVectorStore()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    with pytest.raises(RuntimeError, match="Embedding model failed"):
        retrieval_service.retrieve("Where is my order?")

    assert vector_store.search_calls == []


def test_retrieve_propagates_vector_store_error():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(
        error=RuntimeError("Database search failed")
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    with pytest.raises(RuntimeError, match="Database search failed"):
        retrieval_service.retrieve("Where is my order?")
