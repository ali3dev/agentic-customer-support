
from src.rag.embeddings import EmbeddingService
from src.rag.ingestion import ingest_directory
from src.rag.vector_store import SQLiteVectorStore


def main() -> None:
    embedding_service = EmbeddingService()

    vector_store = SQLiteVectorStore(
        db_path="data/vectors.db"
    )

    result = ingest_directory(
        knowledge_base_path="knowledge_base",
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    print("Knowledge Base Ingestion Complete")
    print("----------------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
