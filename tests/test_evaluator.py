
from src.evaluator import evaluate_response
from src.judge import JudgeResult


class ConfigurableMockJudge:

    def __init__(self, verdicts):
        self.verdicts = verdicts

    def evaluate(
        self,
        behavior_id,
        customer_input,
        assistant_response
    ):
        verdict = self.verdicts.get(
            behavior_id,
            "unknown"
        )

        return JudgeResult(
            verdict=verdict,
            reason=f"Mock judge verdict: {verdict}"
        )


def make_test_case(expected=None, forbidden=None):
    return {
        "id": "TEST-001",
        "input": "My headphones stopped working.",
        "expected_behaviors": expected or [],
        "forbidden_behaviors": forbidden or []
    }


def test_expected_behavior_passes():

    test_case = make_test_case(
        expected=["acknowledge_product_issue"],
        forbidden=["claim_refund_approved"]
    )

    judge = ConfigurableMockJudge({
        "acknowledge_product_issue": "pass",
        "claim_refund_approved": "fail"
    })

    result = evaluate_response(
        test_case,
        "I am sorry your headphones stopped working.",
        judge
    )

    assert result["passed"] is True
    assert result["missing_behaviors"] == []
    assert result["detected_forbidden_behaviors"] == []


def test_missing_expected_behavior_fails():

    test_case = make_test_case(
        expected=["acknowledge_product_issue"]
    )

    judge = ConfigurableMockJudge({
        "acknowledge_product_issue": "fail"
    })

    result = evaluate_response(
        test_case,
        "Please provide your order number.",
        judge
    )

    assert result["passed"] is False
    assert "acknowledge_product_issue" in result["missing_behaviors"]


def test_detected_forbidden_behavior_fails():

    test_case = make_test_case(
        forbidden=["claim_refund_approved"]
    )

    result = evaluate_response(
        test_case,
        "Your refund has been approved."
    )

    assert result["passed"] is False
    assert "claim_refund_approved" in result["detected_forbidden_behaviors"]


def test_judge_confirms_forbidden_behavior_is_absent():

    test_case = make_test_case(
        forbidden=["claim_refund_approved"]
    )

    judge = ConfigurableMockJudge({
        "claim_refund_approved": "fail"
    })

    result = evaluate_response(
        test_case,
        "Your refund request is still under review.",
        judge
    )

    assert result["passed"] is True


def test_missing_judge_produces_unknown():

    test_case = make_test_case(
        expected=["acknowledge_product_issue"]
    )

    result = evaluate_response(
        test_case,
        "I am sorry about the issue."
    )

    assert result["passed"] is False
    assert "acknowledge_product_issue" in result["unknown_behaviors"]


def test_unknown_judge_verdict_does_not_pass():

    test_case = make_test_case(
        expected=["acknowledge_product_issue"]
    )

    judge = ConfigurableMockJudge({
        "acknowledge_product_issue": "unknown"
    })

    result = evaluate_response(
        test_case,
        "I am sorry about the issue.",
        judge
    )

    assert result["passed"] is False
    assert len(result["unknown_behaviors"]) == 1


def test_judge_exception_does_not_crash_evaluator():

    class BrokenJudge:

        def evaluate(
            self,
            behavior_id,
            customer_input,
            assistant_response
        ):
            raise RuntimeError("Simulated judge failure")

    test_case = make_test_case(
        expected=["acknowledge_product_issue"]
    )

    result = evaluate_response(
        test_case,
        "I am sorry about the issue.",
        BrokenJudge()
    )

    assert result["passed"] is False
    assert "acknowledge_product_issue" in result["unknown_behaviors"]
