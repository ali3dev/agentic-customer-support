
import json

from src.llm_client import create_chat_completion
from src.tools.registry import (
    dispatch_tool,
    get_tool_definitions,
)


SYSTEM_PROMPT = """
You are a helpful customer support assistant.

Use available tools when you need order information.
Never invent order status or delivery dates.
Only report information returned by the tools.
If an order cannot be found, explain that clearly.
"""


MAX_ITERATIONS = 3


def build_llm_tools() -> list[dict]:
    """Convert our internal registry into the LLM tool schema."""

    definitions = get_tool_definitions()

    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in definitions
    ]


def run_agent(user_message: str) -> str:
    """Run the LLM tool-calling loop."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    tools = build_llm_tools()

    for _ in range(MAX_ITERATIONS):

        response = create_chat_completion(
            messages=messages,
            tools=tools,
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        # No tool requested: return the final answer.
        if not tool_calls:
            return assistant_message.content or ""

        # Preserve the assistant's tool-call request.
        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        # Execute each requested tool.
        for tool_call in tool_calls:

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                result = dispatch_tool(
                    tool_call.function.name,
                    arguments,
                )

            except (json.JSONDecodeError, TypeError):
                result = {
                    "success": False,
                    "error": "invalid_arguments",
                    "message": "Tool arguments must be valid JSON.",
                }

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return "I couldn't complete your request within the allowed tool-call limit."
