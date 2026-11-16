
import re


RULE_PATTERNS = {
    "claim_refund_approved": [
        r"\b(refund|money back)\b.{0,50}\b(approved|granted|issued)\b",
        r"\b(approved|granted|issued)\b.{0,50}\b(refund|money back)\b",
    ],
    "invent_refund_period": [
        r"\b\d+\s*[- ]?\s*(day|week|month)s?\b.{0,40}\b(refund|return)\b",
        r"\b(refund|return)\b.{0,40}\b\d+\s*[- ]?\s*(day|week|month)s?\b",
    ],
}


NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bno\b",
    r"\bhasn't\b",
    r"\bhaven't\b",
    r"\bwasn't\b",
    r"\bweren't\b",
    r"\bisn't\b",
    r"\baren't\b",
    r"\bnot yet\b",
]


def detect_rule_based(
    behavior: str,
    response: str
) -> bool | None:

    patterns = RULE_PATTERNS.get(behavior)

    if patterns is None:
        return None

    normalized_response = response.lower()

    matches = []

    for pattern in patterns:
        for match in re.finditer(pattern, normalized_response):
            matches.append(match)

    if not matches:
        return False

    for match in matches:
        start = max(0, match.start() - 30)
        end = match.end() + 10

        context = normalized_response[start:end]

        is_negated = any(
            re.search(negation, context)
            for negation in NEGATION_PATTERNS
        )

        if not is_negated:
            return True

    return False