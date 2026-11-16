
from types import SimpleNamespace

from src.agent import run_agent


class FakeAssistantMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        result = {"role": "assistant"}

        if self.content is not None:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]

        return result


def fake_response(message):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )


def make_tool_call(
    call_id="call_123",
    name="get_order_status",
    arguments='{"order_id": "ORD-1001"}',
):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def test_agent_returns_direct_answer(monkeypatch):
    def fake_completion(messages, tools):
        return fake_response(
            FakeAssistantMessage(
                content="Hello! How can I help?"
            )
        )

    monkeypatch.setattr(
        "src.agent.create_chat_completion",
        fake_completion,
    )

    result = run_agent("Hello")

    assert result == "Hello! How can I help?"


def test_agent_executes_tool_and_returns_final_answer(monkeypatch):
    tool_call = make_tool_call()

    responses = iter([
        fake_response(
            FakeAssistantMessage(
                tool_calls=[tool_call]
            )
        ),
        fake_response(
            FakeAssistantMessage(
                content="Your order has been shipped."
            )
        ),
    ])

    captured_messages = []

    def fake_completion(messages, tools):
        captured_messages.append(messages.copy())
        return next(responses)

    monkeypatch.setattr(
        "src.agent.create_chat_completion",
        fake_completion,
    )

    result = run_agent("Where is my order ORD-1001?")

    assert result == "Your order has been shipped."

    tool_messages = [
        message
        for message in captured_messages[1]
        if message["role"] == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_123"
    assert '"shipped"' in tool_messages[0]["content"]


def test_agent_stops_at_iteration_limit(monkeypatch):
    tool_call = make_tool_call(
        call_id="call_repeat"
    )

    call_count = 0

    def fake_completion(messages, tools):
        nonlocal call_count
        call_count += 1

        return fake_response(
            FakeAssistantMessage(
                tool_calls=[tool_call]
            )
        )

    monkeypatch.setattr(
        "src.agent.create_chat_completion",
        fake_completion,
    )

    result = run_agent("Check my order")

    assert call_count == 3
    assert "tool-call limit" in result
    # assert "iteration limit" in result
