from src.tools.schemas import GetOrderStatusInput


# Temporary in-memory order database
ORDERS = {
    "ORD-1001": {
        "status": "shipped",
        "estimated_delivery": "2026-11-18",
    },
    "ORD-1002": {
        "status": "processing",
        "estimated_delivery": None,
    },
    "ORD-1003": {
        "status": "delivered",
        "estimated_delivery": "2026-11-12",
    },
}

def get_order_status(order_id: str) -> dict:
    """Retrieve the current status of a customer order."""
    validated = GetOrderStatusInput(order_id=order_id)

    order = ORDERS.get(validated.order_id)

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
