
import pytest
from pydantic import ValidationError

from src.tools.order_tools import get_order_status


def test_get_existing_order_status():
    result = get_order_status(
        "ORD-1001",
        tenant_id="tenant-demo",
        customer_id="customer-demo",
    )

    assert result["success"] is True
    assert result["status"] == "shipped"
    assert result["order_id"] == "ORD-1001"


def test_get_nonexistent_order():
    result = get_order_status(
        "ORD-9999",
        tenant_id="tenant-demo",
        customer_id="customer-demo",
    )

    assert result["success"] is False
    assert result["error"] == "order_not_found"


def test_reject_invalid_order_id():
    with pytest.raises(ValidationError):
        get_order_status(
            "A",
            tenant_id="tenant-demo",
            customer_id="customer-demo",
        )


def test_order_lookup_requires_matching_tenant_and_customer():
    tenant_mismatch = get_order_status(
        "ORD-1001",
        tenant_id="other-tenant",
        customer_id="customer-demo",
    )
    customer_mismatch = get_order_status(
        "ORD-1001",
        tenant_id="tenant-demo",
        customer_id="other-customer",
    )
    missing_identity = get_order_status(
        "ORD-1001",
        tenant_id=None,
        customer_id=None,
    )

    assert tenant_mismatch["error"] == "order_not_found"
    assert customer_mismatch["error"] == "order_not_found"
    assert missing_identity["error"] == "order_not_found"
