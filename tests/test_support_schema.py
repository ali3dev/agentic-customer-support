import pytest
from pydantic import ValidationError

from src.schemas.support import SupportResponse


def test_valid_support_response():
    response = SupportResponse(
        status="answered",
        message="Your order is being investigated.",
        needs_human_review=False,
    )

    assert response.status == "answered"
    assert response.needs_human_review is False


def test_invalid_status_is_rejected():
    with pytest.raises(ValidationError):
        SupportResponse(
            status="pending",
            message="Your order is being investigated.",
            needs_human_review=False,
        )


def test_string_boolean_is_rejected():
    with pytest.raises(ValidationError):
        SupportResponse(
            status="answered",
            message="Your order is being investigated.",
            needs_human_review="false",
        )


def test_missing_field_is_rejected():
    with pytest.raises(ValidationError):
        SupportResponse(
            status="answered",
            message="Your order is being investigated.",
        )