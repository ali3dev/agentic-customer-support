
from src.tools.registry import (
    dispatch_tool,
    get_tool_definitions,
)


def test_tool_definition_is_available():
    definitions = get_tool_definitions()

    assert len(definitions) == 1
    assert definitions[0]["name"] == "get_order_status"


def test_dispatch_existing_tool():
    result = dispatch_tool(
        "get_order_status",
        {"order_id": "ORD-1001"},
    )

    assert result["success"] is True
    assert result["status"] == "shipped"


def test_dispatch_unknown_tool():
    result = dispatch_tool(
        "delete_customer",
        {},
    )

    assert result["success"] is False
    assert result["error"] == "unknown_tool"


def test_dispatch_invalid_arguments():
    result = dispatch_tool(
        "get_order_status",
        {"order_id": "A"},
    )

    assert result["success"] is False
    assert result["error"] == "invalid_arguments"
