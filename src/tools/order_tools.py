from src.tools.schemas import GetOrderStatusInput


# Demo orders are keyed by authenticated tenant and customer identities.
ORDERS = {
    ("tenant-demo", "customer-demo", "ORD-1001"): {
        "status": "shipped",
        "estimated_delivery": "2026-11-18",
    },
    ("tenant-demo", "customer-demo", "ORD-1002"): {
        "status": "processing",
        "estimated_delivery": None,
    },
    ("tenant-demo", "customer-demo", "ORD-1003"): {
        "status": "delivered",
        "estimated_delivery": "2026-11-12",
    },
}


def get_order_status(
    order_id: str,
    *,
    tenant_id: str | None,
    customer_id: str | None,
) -> dict:
    """Retrieve the current status of a customer order."""
    validated = GetOrderStatusInput(order_id=order_id)

    if (
        not isinstance(tenant_id, str)
        or not tenant_id.strip()
        or not isinstance(customer_id, str)
        or not customer_id.strip()
    ):
        return {
            "success": False,
            "error": "order_not_found",
            "message": "No order found with this ID.",
        }

    order = ORDERS.get(
        (tenant_id, customer_id, validated.order_id)
    )

    if order is None:
        return {
            "success": False,
            "error": "order_not_found",
            "message": "No order found with this ID.",
        }

    return {
        "success": True,
        "order_id": validated.order_id,
        "status": order["status"],
        "estimated_delivery": order["estimated_delivery"],
    }
