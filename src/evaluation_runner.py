
import json
from pathlib import Path

from src.evaluator import evaluate_response
from src.judge import BehaviorJudge


def load_golden_cases(file_path: str | Path) -> list[dict]:
    """
    Load test cases from a JSON file.
    """
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not isinstance(test_cases, list):
        raise ValueError("Golden test data must be a JSON list.")

    return test_cases


def classify_result(result: dict) -> str:
    """
    Classify an evaluation result as PASS, FAIL, or INCOMPLETE.
    """

    if result["passed"]:
        return "PASS"

    has_definite_failure = bool(
        result["missing_behaviors"]
        or result["detected_forbidden_behaviors"]
    )

    if has_definite_failure:
        return "FAIL"

    if result["unknown_behaviors"]:
        return "INCOMPLETE"

    return "FAIL"


def run_evaluations(
    test_cases: list[dict],
    responses: dict[str, str],
    judge: BehaviorJudge | None = None
) -> dict:
    """
    Evaluate all test cases using their corresponding responses.
    """

    results = []

    for test_case in test_cases:
        test_id = test_case["id"]

        if test_id not in responses:
            raise ValueError(
                f"Missing assistant response for test case: {test_id}"
            )

        result = evaluate_response(
            test_case=test_case,
            actual_response=responses[test_id],
            judge=judge
        )

        results.append(result)

    passed = sum(
        classify_result(result) == "PASS"
        for result in results
    )

    failed = sum(
        classify_result(result) == "FAIL"
        for result in results
    )

    incomplete = sum(
        classify_result(result) == "INCOMPLETE"
        for result in results
    )

    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "incomplete": incomplete,
        "results": results
    }


def print_report(report: dict) -> None:
    """
    Display a readable evaluation report.
    """

    print("\n========== EVALUATION REPORT ==========")

    print(f"Total tests: {report['total']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Incomplete: {report['incomplete']}")

    print("\n------------- DETAILS ----------------")

    for result in report["results"]:
        status = classify_result(result)

        print(f"{result['test_id']}: {status}")
        print(f"Reason: {result['reason']}")

    print("======================================\n")
