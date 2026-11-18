
from typing import Any

from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import SQLiteVectorStore


DEFAULT_RELEVANCE_THRESHOLD = 0.6


class RetrievalService:
    """Retrieve relevant knowledge-base chunks for customer queries."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: SQLiteVectorStore,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> None:
        if (
            isinstance(relevance_threshold, bool)
            or not isinstance(relevance_threshold, (int, float))
            or not -1.0 <= relevance_threshold <= 1.0
        ):
            raise ValueError(
                "relevance_threshold must be between -1 and 1."
            )

        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Embed a customer query and retrieve its most relevant chunks."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        query_embedding = self.embedding_service.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return [
            result
            for result in results
            if result["score"] >= self.relevance_threshold
        ]
