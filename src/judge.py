
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class JudgeResult:
    """
    Result of checking whether a behavior is present.

    pass    = behavior is present
    fail    = behavior is absent
    unknown = judge cannot decide
    """

    verdict: Literal["pass", "fail", "unknown"]
    reason: str


class BehaviorJudge(Protocol):
    """
    Contract that every behavior judge must implement.
    """

    def evaluate(
        self,
        behavior_id: str,
        customer_input: str,
        assistant_response: str
    ) -> JudgeResult:
        """
        Evaluate whether the requested behavior is present.
        """
        ...
