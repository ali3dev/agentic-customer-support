from types import SimpleNamespace

from src.llm_client import DEFAULT_LLM_TIMEOUT_MS, create_chat_completion


def test_chat_completion_uses_configured_model_and_timeout(monkeypatch):
    observed = {}
    expected_response = object()

    class FakeOpenRouter:
        def __init__(self, **kwargs):
            observed["client"] = kwargs
            self.chat = SimpleNamespace(
                send=self.send,
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def send(self, **kwargs):
            observed["request"] = kwargs
            return expected_response

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setattr("src.llm_client.OpenRouter", FakeOpenRouter)

    response = create_chat_completion(messages=[{"role": "user", "content": "Hi"}])

    assert response is expected_response
    assert observed["client"]["timeout_ms"] == DEFAULT_LLM_TIMEOUT_MS
    assert observed["request"]["timeout_ms"] == DEFAULT_LLM_TIMEOUT_MS
    assert observed["request"]["model"] == "anthropic/claude-sonnet-4.6"
    assert observed["request"]["messages"] == [
        {"role": "user", "content": "Hi"}
    ]
