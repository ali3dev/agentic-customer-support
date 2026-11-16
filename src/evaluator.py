
from src.rules import detect_rule_based
from src.judge import BehaviorJudge


def _detect_behavior(
    behavior_id: str,
    customer_input: str,
    assistant_response: str,
    judge: BehaviorJudge | None
) -> tuple[bool | None, str, str]:
    """
    Determine whether a behavior is present.

    Returns:
        True  = behavior is present
        False = behavior is absent
        None  = behavior could not be evaluated
    """

    # Step 1: Try rule-based detection.
    rule_result = detect_rule_based(
        behavior_id,
        assistant_response
    )

    # A positive rule match is evidence that the behavior exists.
    if rule_result is True:
        return (
            True,
            "rules",
            "Behavior detected by a rule-based pattern."
        )

    # Step 2: If rules did not detect it, use the semantic judge.
    if judge is None:
        return (
            None,
            "unavailable",
            "No positive rule match and no judge was provided."
        )

    try:
        judge_result = judge.evaluate(
            behavior_id=behavior_id,
            customer_input=customer_input,
            assistant_response=assistant_response
        )

    except Exception as error:
        return (
            None,
            "judge_error",
            f"Judge evaluation failed: {type(error).__name__}."
        )

    # Step 3: Interpret the judge's verdict.
    if judge_result.verdict == "pass":
        return (
            True,
            "judge",
            judge_result.reason
        )

    if judge_result.verdict == "fail":
        return (
            False,
            "judge",
            judge_result.reason
        )

    if judge_result.verdict == "unknown":
        return (
            None,
            "judge",
            judge_result.reason
        )

    # Protect against an invalid judge response.
    return (
        None,
        "judge_error",
        "Judge returned an invalid verdict."
    )


def evaluate_response(
    test_case: dict,
    actual_response: str,
    judge: BehaviorJudge | None = None
) -> dict:
    """
    Evaluate an AI response against a golden test case.

    Expected behavior:
        The behavior must be present.

    Forbidden behavior:
        The behavior must be absent.

    An unknown result never silently passes.
    """

    result = {
        "test_id": test_case["id"],
        "passed": False,
        "missing_behaviors": [],
        "detected_forbidden_behaviors": [],
        "unknown_behaviors": [],
        "evaluation_details": [],
        "reason": ""
    }

    customer_input = test_case.get("input", "")

    # Evaluate expected behaviors.
    for behavior in test_case["expected_behaviors"]:

        detected, source, reason = _detect_behavior(
            behavior_id=behavior,
            customer_input=customer_input,
            assistant_response=actual_response,
            judge=judge
        )

        result["evaluation_details"].append({
            "behavior": behavior,
            "type": "expected",
            "detected": detected,
            "source": source,
            "reason": reason
        })

        if detected is False:
            result["missing_behaviors"].append(behavior)

        elif detected is None:
            result["unknown_behaviors"].append(behavior)

    # Evaluate forbidden behaviors.
    for behavior in test_case["forbidden_behaviors"]:

        detected, source, reason = _detect_behavior(
            behavior_id=behavior,
            customer_input=customer_input,
            assistant_response=actual_response,
            judge=judge
        )

        result["evaluation_details"].append({
            "behavior": behavior,
            "type": "forbidden",
            "detected": detected,
            "source": source,
            "reason": reason
        })

        if detected is True:
            result["detected_forbidden_behaviors"].append(behavior)

        elif detected is None:
            result["unknown_behaviors"].append(behavior)

    # Final decision.
    result["passed"] = (
        len(result["missing_behaviors"]) == 0
        and len(result["detected_forbidden_behaviors"]) == 0
        and len(result["unknown_behaviors"]) == 0
    )

    if result["passed"]:
        result["reason"] = "All evaluation checks passed."

    elif result["unknown_behaviors"]:
        result["reason"] = (
            "Evaluation incomplete: some behaviors "
            "could not be verified."
        )

    else:
        result["reason"] = (
            "One or more evaluation checks failed."
        )

    return result
