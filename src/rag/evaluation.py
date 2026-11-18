from typing import Any


def evaluate_retrieval(
    retrieval_service: Any,
    test_cases: list[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    """Measure source recall at k for a labeled retrieval test set."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    results = []
    total_recall = 0.0

    for test_case in test_cases:
        if not isinstance(test_case, dict):
            raise ValueError("Every retrieval test case must be a dictionary.")

        query = test_case.get("query")
        expected_sources = test_case.get("expected_source_paths")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Every test case must have a non-empty query.")

        if (
            not isinstance(expected_sources, list)
            or not expected_sources
            or any(
                not isinstance(source, str) or not source.strip()
                for source in expected_sources
            )
        ):
            raise ValueError(
                "Every test case must have non-empty expected_source_paths."
            )

        retrieved = retrieval_service.retrieve(query, top_k=top_k)
        retrieved_sources = {
            item["source_path"]
            for item in retrieved
        }
        expected = set(expected_sources)
        recall = len(expected & retrieved_sources) / len(expected)
        total_recall += recall

        results.append({
            "id": test_case.get("id"),
            "query": query,
            "expected_source_paths": sorted(expected),
            "retrieved_source_paths": sorted(retrieved_sources),
            "recall_at_k": recall,
            "passed": recall == 1.0,
        })

    total = len(results)
    passed = sum(result["passed"] for result in results)

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "recall_at_k": total_recall / total if total else 0.0,
        "results": results,
    }
