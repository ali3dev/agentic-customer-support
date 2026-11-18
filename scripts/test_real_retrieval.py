
from src.rag.embeddings import EmbeddingService
from src.rag.retrieval import RetrievalService
from src.rag.vector_store import SQLiteVectorStore


def main() -> None:
    embedding_service = EmbeddingService()

    vector_store = SQLiteVectorStore(
        db_path="data/vectors.db"
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    queries = [
        "Can I return an unused product?",
        "How long does a refund take?",
        "My product arrived damaged. What should I do?",
        "How long does standard shipping take?",
        "How can I track my order?",
        "Do you accept returns for digital products?",
    ]

    for index, query in enumerate(queries, start=1):
        print("\n" + "=" * 80)
        print(f"QUERY {index}: {query}")
        print("=" * 80)

        results = retrieval_service.retrieve(
            query=query,
            top_k=3,
        )

        if not results:
            print("No matching chunks found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['source_path']}")
            print(f"Section: {result['section_title']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()
