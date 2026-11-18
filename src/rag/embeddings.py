
import threading

from fastembed import TextEmbedding


class EmbeddingService:
    """Generate vector embeddings for documents and queries."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.model = TextEmbedding(model_name=model_name)
        self._lock = threading.Lock()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Convert multiple documents into embedding vectors."""

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Document texts cannot be empty.")

        with self._lock:
            return [
                embedding.tolist()
                for embedding in self.model.embed(texts)
            ]

    def embed_query(self, text: str) -> list[float]:
        """Convert one customer query into an embedding vector."""

        if not text.strip():
            raise ValueError("Query text cannot be empty.")

        with self._lock:
            embedding = next(
                self.model.query_embed(text)
            )

        return embedding.tolist()
