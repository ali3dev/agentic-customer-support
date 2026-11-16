
import pytest
from pydantic import ValidationError

from src.tools.order_tools import get_order_status


def test_get_existing_order_status():
    result = get_order_status("ORD-1001")

    assert result["success"] is True
    assert result["status"] == "shipped"
    assert result["order_id"] == "ORD-1001"


def test_get_nonexistent_order():
    result = get_order_status("ORD-9999")

    assert result["success"] is False
    assert result["error"] == "order_not_found"


def test_reject_invalid_order_id():
    with pytest.raises(ValidationError):
        get_order_status("A")
