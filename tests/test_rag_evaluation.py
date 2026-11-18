import json
from pathlib import Path

import pytest

from src.rag.evaluation import evaluate_retrieval


class FakeRetrievalService:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, [])[:top_k]


def test_golden_retrieval_cases_have_expected_sources():
    cases_path = Path(__file__).with_name("rag_golden_cases.json")
    test_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    retrieval_service = FakeRetrievalService({
        "How long can I return an unused product?": [
            {"source_path": "policies/returns.md"},
        ],
        "How long does standard shipping take?": [
            {"source_path": "policies/shipping.md"},
        ],
        "How do I track an order?": [
            {"source_path": "faqs/general.md"},
        ],
    })

    report = evaluate_retrieval(retrieval_service, test_cases, top_k=3)

    assert report["total"] == 3
    assert report["passed"] == 3
    assert report["failed"] == 0
    assert report["recall_at_k"] == 1.0
    assert retrieval_service.calls == [
        (case["query"], 3)
        for case in test_cases
    ]


def test_retrieval_evaluation_reports_missing_expected_sources():
    retrieval_service = FakeRetrievalService({
        "return policy": [
            {"source_path": "policies/shipping.md"},
        ],
    })

    report = evaluate_retrieval(
        retrieval_service,
        [{
            "id": "RAG-miss",
            "query": "return policy",
            "expected_source_paths": ["policies/returns.md"],
        }],
    )

    assert report["passed"] == 0
    assert report["failed"] == 1
    assert report["recall_at_k"] == 0.0


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_retrieval_evaluation_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_retrieval(FakeRetrievalService({}), [], top_k=top_k)


def test_retrieval_evaluation_rejects_invalid_test_cases():
    with pytest.raises(ValueError, match="expected_source_paths"):
        evaluate_retrieval(
            FakeRetrievalService({}),
            [{"query": "return policy", "expected_source_paths": []}],
        )
