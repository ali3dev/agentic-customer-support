
import pytest

from src.evaluation_runner import (
    classify_result,
    load_golden_cases,
    run_evaluations,
)
from src.judge import JudgeResult


class ConfigurableMockJudge:
    def __init__(self, verdicts):
        self.verdicts = verdicts

    def evaluate(self, behavior_id, customer_input, assistant_response):
        verdict = self.verdicts.get(behavior_id, "unknown")
        return JudgeResult(
            verdict=verdict,
            reason=f"Mock verdict: {verdict}"
        )


def make_result(
    passed=False,
    missing=None,
    forbidden=None,
    unknown=None
):
    return {
        "passed": passed,
        "missing_behaviors": missing or [],
        "detected_forbidden_behaviors": forbidden or [],
        "unknown_behaviors": unknown or [],
    }


def test_classify_passed_result():
    result = make_result(passed=True)

    assert classify_result(result) == "PASS"


def test_classify_definite_failure():
    result = make_result(
        missing=["acknowledge_product_issue"]
    )

    assert classify_result(result) == "FAIL"


def test_classify_detected_forbidden_behavior():
    result = make_result(
        forbidden=["claim_refund_approved"]
    )

    assert classify_result(result) == "FAIL"


def test_classify_unknown_result():
    result = make_result(
        unknown=["ask_for_purchase_details"]
    )

    assert classify_result(result) == "INCOMPLETE"


def test_definite_failure_takes_priority_over_unknown():
    result = make_result(
        missing=["acknowledge_product_issue"],
        unknown=["ask_for_purchase_details"]
    )

    assert classify_result(result) == "FAIL"



def test_run_evaluations_generates_correct_summary():
    test_cases = [
        {
            "id": "TC-PASS",
            "input": "My headphones stopped working.",
            "expected_behaviors": [
                "acknowledge_product_issue"
            ],
            "forbidden_behaviors": []
        },
        {
            "id": "TC-FAIL",
            "input": "My headphones stopped working.",
            "expected_behaviors": [
                "acknowledge_product_issue",
                "ask_for_purchase_details"
            ],
            "forbidden_behaviors": []
        },
        {
            "id": "TC-INCOMPLETE",
            "input": "My order is delayed.",
            "expected_behaviors": [
                "explain_tracking_capability_is_unavailable"
            ],
            "forbidden_behaviors": []
        }
    ]

    responses = {
        "TC-PASS": "I am sorry your headphones stopped working.",
        "TC-FAIL": "I am sorry about your issue.",
        "TC-INCOMPLETE": "I will look into your order."
    }

    judge = ConfigurableMockJudge({
        "acknowledge_product_issue": "pass",
        "ask_for_purchase_details": "fail",
        "explain_tracking_capability_is_unavailable": "unknown"
    })

    report = run_evaluations(
        test_cases=test_cases,
        responses=responses,
        judge=judge
    )

    assert report["total"] == 3
    assert report["passed"] == 1
    assert report["failed"] == 1
    assert report["incomplete"] == 1



def test_missing_response_raises_error():
    test_cases = [
        {
            "id": "TC-001",
            "input": "Where is my order?",
            "expected_behaviors": [],
            "forbidden_behaviors": []
        }
    ]

    with pytest.raises(ValueError, match="Missing assistant response"):
        run_evaluations(
            test_cases=test_cases,
            responses={}
        )


def test_load_golden_cases():
    cases = load_golden_cases("tests/golden_cases.json")

    assert len(cases) == 3
    assert cases[0]["id"] == "TC-001"

