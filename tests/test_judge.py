from src.judge import JudgeResult, BehaviorJudge


class MockJudge:

    def evaluate(
        self,
        behavior_id: str,
        customer_input: str,
        assistant_response: str
    ) -> JudgeResult:

        return JudgeResult(
            verdict="pass",
            reason="Mock evaluation successful"
        )


def test_mock_judge_returns_valid_result():

    judge: BehaviorJudge = MockJudge()

    result = judge.evaluate(
        behavior_id="acknowledge_product_issue",
        customer_input="My headphones stopped working.",
        assistant_response="I am sorry your headphones stopped working."
    )

    assert result.verdict == "pass"
    assert result.reason == "Mock evaluation successful"