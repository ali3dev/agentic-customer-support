import os
from typing import Any

from dotenv import load_dotenv
from openrouter import OpenRouter


load_dotenv()


def create_chat_completion(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> Any:
    """Send a chat completion request to OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing.")

    if not model:
        raise RuntimeError("OPENROUTER_MODEL is missing.")

    request = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.2,
        "max_completion_tokens": 128
    }

    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"

    with OpenRouter(api_key=api_key) as client:
        response = client.chat.send(**request)

    return response