from src.rules import detect_rule_based


def test_detect_approved_refund():
    result = detect_rule_based(
        "claim_refund_approved",
        "Your refund has been approved."
    )

    assert result is True


def test_detect_refund_period():
    result = detect_rule_based(
        "invent_refund_period",
        "Our refund policy allows returns within 30 days."
    )

    assert result is True


def test_unknown_behavior_returns_none():
    result = detect_rule_based(
        "acknowledge_product_issue",
        "I am sorry your headphones stopped working."
    )

    assert result is None


def test_unrelated_response_does_not_trigger_rule():
    result = detect_rule_based(
        "claim_refund_approved",
        "I am sorry your headphones stopped working."
    )

    assert result is False

def test_not_approved_refund_does_not_trigger_rule():
    result = detect_rule_based(
        "claim_refund_approved",
        "Your refund has not been approved yet."
    )

    assert result is False

def test_not_granted_refund_does_not_trigger_rule():
    result = detect_rule_based(
        "claim_refund_approved",
        "Your refund has not been granted."
    )

    assert result is False


def test_approved_refund_with_different_word_order():
    result = detect_rule_based(
        "claim_refund_approved",
        "We have approved your refund."
    )

    assert result is True


def test_negation_without_approval_claim():
    result = detect_rule_based(
        "claim_refund_approved",
        "Your refund request is still under review."
    )

    assert result is False