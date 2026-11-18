import json
from pathlib import Path

from src.rag.embeddings import EmbeddingService
from src.rag.evaluation import evaluate_retrieval
from src.rag.retrieval import RetrievalService
from src.rag.vector_store import SQLiteVectorStore


def main() -> None:
    test_cases_path = Path(__file__).parents[1] / "tests" / "rag_golden_cases.json"
    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))

    retrieval_service = RetrievalService(
        embedding_service=EmbeddingService(),
        vector_store=SQLiteVectorStore("data/vectors.db"),
    )

    report = evaluate_retrieval(
        retrieval_service,
        test_cases,
        top_k=3,
    )

    print(json.dumps(report, indent=2))

    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
