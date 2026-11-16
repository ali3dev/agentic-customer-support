from pydantic import ValidationError

from src.tools.order_tools import get_order_status
from src.tools.schemas import GetOrderStatusInput


TOOL_REGISTRY = {
    "get_order_status": {
        "description": (
            "Retrieve the current status of a customer order "
            "using its unique order ID."
        ),
        "input_model": GetOrderStatusInput,
        "function": get_order_status,
    }
}

def get_tool_definitions() -> list[dict]:

    definitions = []

    for name, tool in TOOL_REGISTRY.items():
        definitions.append({
            "name": name,
            "description": tool["description"],
            "parameters": tool["input_model"].model_json_schema(),
        })

    return definitions


def dispatch_tool(name: str, arguments: dict) -> dict:

    tool = TOOL_REGISTRY.get(name)

    if tool is None:
        return {
            "success": False,
            "error": "unknown_tool",
        }

    if not isinstance(arguments, dict):
        return {
            "success": False,
            "error": "invalid_arguments",
        }

    try:
        return tool["function"](**arguments)

    except ValidationError as exc:
        return {
            "success": False,
            "error": "invalid_arguments",
            "details": [
                {
                    "field": ".".join(str(part) for part in error["loc"]),
                    "message": error["msg"],
                }
                for error in exc.errors()
            ],
        }