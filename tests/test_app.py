import json
import hashlib
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openrouter.errors import OpenRouterError
from requests.exceptions import ConnectionError
from redis.exceptions import RedisError

from src.app import AppSettings, create_app
from src.agent import run_agent_response
from src.schemas.support import AgentResponse, SourceMetadata, TokenUsage
from src.security import Principal
from src.tools.order_tools import ORDERS


TENANT_A_CUSTOMER_KEY = "tenant-a-customer-test-key-00000001"
TENANT_A_ADMIN_KEY = "tenant-a-admin-test-key-000000000001"
TENANT_B_CUSTOMER_KEY = "tenant-b-customer-test-key-00000001"
TENANT_B_ADMIN_KEY = "tenant-b-admin-test-key-000000000001"
PLATFORM_ADMIN_KEY = "platform-admin-test-key-0000000000001"
TENANT_A_OTHER_CUSTOMER_KEY = "tenant-a-other-customer-test-key-00001"


class FakeEmbeddingService:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def app_factory(tmp_path):
    agent_calls = []

    def fake_agent(
        message,
        retrieval_service,
        history,
        allowed_tool_names,
        tenant_id=None,
        customer_id=None,
    ):
        agent_calls.append({
            "message": message,
            "retrieval_service": retrieval_service,
            "history": history,
            "allowed_tool_names": allowed_tool_names,
            "tenant_id": tenant_id,
            "customer_id": customer_id,
        })
        return AgentResponse(
            message=f"Support reply: {message}",
            sources=[
                SourceMetadata(
                    citation_id="S1",
                    chunk_id="chunk-1",
                    source_path="policies/returns.md",
                    section_title="Returns",
                    score=0.91,
                )
            ],
            usage=TokenUsage(
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
                cost_usd=0.0003,
            ),
        )

    credentials = {
        TENANT_A_CUSTOMER_KEY: Principal(
            "tenant-a",
            "customer",
            "customer-a",
        ),
        TENANT_A_ADMIN_KEY: Principal("tenant-a", "tenant_admin"),
        TENANT_A_OTHER_CUSTOMER_KEY: Principal(
            "tenant-a",
            "customer",
            "customer-a-other",
        ),
        TENANT_B_CUSTOMER_KEY: Principal(
            "tenant-b",
            "customer",
            "customer-b",
        ),
        TENANT_B_ADMIN_KEY: Principal("tenant-b", "tenant_admin"),
        PLATFORM_ADMIN_KEY: Principal("platform", "platform_admin"),
    }

    def create(rate_limit_requests=100, agent_runner_override=None):
        app = create_app(
            settings=AppSettings(
                initial_api_keys=credentials,
                data_dir=tmp_path / "data",
                knowledge_base_dir=tmp_path / "knowledge",
                rate_limit_requests=rate_limit_requests,
                rate_limit_window_seconds=60,
            ),
            embedding_service_factory=FakeEmbeddingService,
            agent_runner=agent_runner_override or fake_agent,
        )
        return app

    return create, agent_calls, tmp_path


def test_health_dashboard_auth_and_request_validation(app_factory):
    create_app, _, _ = app_factory

    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/dashboard").status_code == 200
        assert client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        ).status_code == 401
        assert client.post(
            "/api/v1/chat",
            headers={"X-API-Key": TENANT_A_CUSTOMER_KEY},
            json={"message": "", "unexpected": True},
        ).status_code == 422
        assert client.post(
            "/api/v1/chat",
            headers={"X-API-Key": TENANT_A_CUSTOMER_KEY},
            json={
                "message": "Check my order",
                "tenant_id": "tenant-b",
                "customer_id": "customer-b",
            },
        ).status_code == 422


def test_app_refuses_startup_without_platform_admin(tmp_path):
    app = create_app(
        settings=AppSettings(
            initial_api_keys={
                TENANT_A_CUSTOMER_KEY: Principal(
                    "tenant-a",
                    "customer",
                    "customer-a",
                ),
            },
            data_dir=tmp_path / "data",
            knowledge_base_dir=tmp_path / "knowledge",
        ),
        embedding_service_factory=FakeEmbeddingService,
    )

    with pytest.raises(RuntimeError, match="platform_admin API key"):
        with TestClient(app):
            pass


def test_chat_persists_history_and_isolates_tenants(app_factory):
    create_app, agent_calls, _ = app_factory

    with TestClient(create_app()) as client:
        headers = {"X-API-Key": TENANT_A_CUSTOMER_KEY}
        first = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "First question"},
        )
        assert first.status_code == 200
        assert agent_calls[0]["tenant_id"] == "tenant-a"
        assert agent_calls[0]["customer_id"] == "customer-a"
        first_payload = first.json()
        conversation_id = first_payload["conversation_id"]
        assert first_payload["sources"][0]["source_path"] == (
            "policies/returns.md"
        )
        assert first_payload["usage"]["total_tokens"] == 20

        second = client.post(
            "/api/v1/support",
            headers=headers,
            json={
                "message": "Follow-up question",
                "conversation_id": conversation_id,
            },
        )
        assert second.status_code == 200
        assert agent_calls[1]["history"] == [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Support reply: First question"},
        ]

        messages = client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=headers,
        )
        assert messages.status_code == 200
        assert len(messages.json()["messages"]) == 4
        assert messages.json()["messages"][1]["sources"][0]["citation_id"] == "S1"
        other_customer_headers = {
            "X-API-Key": TENANT_A_OTHER_CUSTOMER_KEY,
        }
        assert client.get(
            "/api/v1/conversations",
            headers=other_customer_headers,
        ).json() == []
        assert client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=other_customer_headers,
        ).status_code == 404
        assert client.post(
            "/api/v1/chat",
            headers=other_customer_headers,
            json={
                "message": "Read another customer's conversation",
                "conversation_id": conversation_id,
            },
        ).status_code == 404
        assert client.delete(
            f"/api/v1/conversations/{conversation_id}",
            headers=other_customer_headers,
        ).status_code == 404

        admin_headers = {"X-API-Key": TENANT_A_ADMIN_KEY}
        assert any(
            item["conversation_id"] == conversation_id
            for item in client.get(
                "/api/v1/conversations",
                headers=admin_headers,
            ).json()
        )
        assert client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=admin_headers,
        ).status_code == 200

        assert client.get(
            "/api/v1/admin/usage",
            headers=admin_headers,
        ).json() == {
            "prompt_tokens": 24,
            "completion_tokens": 16,
            "total_tokens": 40,
            "cost_usd": 0.0006,
            "cost_records": 2,
        }
        assert client.delete(
            f"/api/v1/conversations/{conversation_id}",
            headers=headers,
        ).status_code == 204
        assert client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=headers,
        ).status_code == 404

        tenant_b_headers = {"X-API-Key": TENANT_B_CUSTOMER_KEY}
        assert client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=tenant_b_headers,
        ).status_code == 404
        assert client.get(
            "/api/v1/conversations",
            headers=tenant_b_headers,
        ).json() == []

        assert client.delete(
            f"/api/v1/conversations/{conversation_id}",
            headers=tenant_b_headers,
        ).status_code == 404


def test_legacy_customer_without_identity_cannot_use_conversations(
    app_factory,
    monkeypatch,
):
    create_app, _, _ = app_factory
    app = create_app()

    with TestClient(app) as client:
        monkeypatch.setattr(
            app.state.api_keys,
            "authenticate",
            lambda _: Principal("tenant-a", "customer"),
        )
        headers = {"X-API-Key": "legacy-customer-key"}
        assert client.get(
            "/api/v1/conversations",
            headers=headers,
        ).status_code == 403
        assert client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "Continue a conversation"},
        ).status_code == 403


def test_order_tool_checks_authenticated_tenant_and_customer(
    app_factory,
    monkeypatch,
):
    create_app, _, _ = app_factory
    monkeypatch.setitem(
        ORDERS,
        ("tenant-a", "customer-a", "ORD-1001"),
        {
            "status": "shipped",
            "estimated_delivery": "2026-11-18",
        },
    )

    def fake_completion(messages, tools):
        if messages[-1]["role"] == "tool":
            tool_result = json.loads(messages[-1]["content"])
            if tool_result["success"]:
                content = f"Order status is {tool_result['status']}."
            else:
                content = "No order found."
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=content,
                            tool_calls=[],
                        )
                    )
                ]
            )

        tool_call = SimpleNamespace(
            id="call-order",
            type="function",
            function=SimpleNamespace(
                name="get_order_status",
                arguments='{"order_id":"ORD-1001"}',
            ),
        )

        class AssistantToolMessage:
            tool_calls = [tool_call]
            content = None

            def model_dump(self, exclude_none=True):
                return {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }],
                }

        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=AssistantToolMessage())
            ]
        )

    monkeypatch.setattr(
        "src.agent.create_chat_completion",
        fake_completion,
    )

    with TestClient(create_app(agent_runner_override=run_agent_response)) as client:
        owner_response = client.post(
            "/api/v1/chat",
            headers={"X-API-Key": TENANT_A_CUSTOMER_KEY},
            json={"message": "Check ORD-1001"},
        )
        other_customer_response = client.post(
            "/api/v1/chat",
            headers={"X-API-Key": TENANT_A_OTHER_CUSTOMER_KEY},
            json={"message": "Check ORD-1001"},
        )
        other_tenant_response = client.post(
            "/api/v1/chat",
            headers={"X-API-Key": TENANT_B_CUSTOMER_KEY},
            json={"message": "Check ORD-1001"},
        )

    assert "shipped" in owner_response.json()["message"]
    assert other_customer_response.json()["message"] == "No order found."
    assert other_tenant_response.json()["message"] == "No order found."


def test_conversation_survives_application_restart(app_factory):
    create_app, _, _ = app_factory
    headers = {"X-API-Key": TENANT_A_CUSTOMER_KEY}

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "Persist this conversation"},
        )
        conversation_id = response.json()["conversation_id"]

    with TestClient(create_app()) as restarted_client:
        restored = restarted_client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=headers,
        )

    assert restored.status_code == 200
    assert restored.json()["messages"][0]["content"] == (
        "Persist this conversation"
    )


def test_tenant_admin_manages_only_its_knowledge_base(app_factory):
    create_app, _, tmp_path = app_factory
    tenant_kb_root = tmp_path / "knowledge"

    with TestClient(create_app()) as client:
        customer_headers = {"X-API-Key": TENANT_A_CUSTOMER_KEY}
        assert client.put(
            "/api/v1/knowledge-base/returns.md",
            headers=customer_headers,
            json={"content": "# Returns\n\nReturns within 30 days."},
        ).status_code == 403

        admin_headers = {"X-API-Key": TENANT_A_ADMIN_KEY}
        response = client.put(
            "/api/v1/knowledge-base/returns.md",
            headers=admin_headers,
            json={"content": "# Returns\n\nReturns within 30 days."},
        )
        assert response.status_code == 200
        assert response.json()["ingestion"]["chunks_stored"] == 1
        assert client.get(
            "/api/v1/knowledge-base",
            headers=admin_headers,
        ).json() == [{"name": "returns.md", "path": "returns.md"}]
        assert len(list(tenant_kb_root.rglob("returns.md"))) == 1

        tenant_b_docs = client.get(
            "/api/v1/knowledge-base",
            headers={"X-API-Key": TENANT_B_CUSTOMER_KEY},
        )
        assert tenant_b_docs.status_code == 403
        tenant_b_admin_headers = {"X-API-Key": TENANT_B_ADMIN_KEY}
        assert client.get(
            "/api/v1/knowledge-base",
            headers=tenant_b_admin_headers,
        ).json() == []

        assert client.put(
            "/api/v1/knowledge-base/shipping.md",
            headers=tenant_b_admin_headers,
            json={"content": "# Shipping\n\nShipping is tracked."},
        ).status_code == 200
        assert len(list(tenant_kb_root.rglob("*.md"))) == 2
        assert client.get(
            "/api/v1/knowledge-base",
            headers=admin_headers,
        ).json() == [{"name": "returns.md", "path": "returns.md"}]
        assert client.delete(
            "/api/v1/knowledge-base/returns.md",
            headers=admin_headers,
        ).status_code == 200
        assert client.get(
            "/api/v1/knowledge-base",
            headers=admin_headers,
        ).json() == []
        assert len(list(tenant_kb_root.rglob("*.md"))) == 1
        assert client.get(
            "/api/v1/knowledge-base",
            headers=tenant_b_admin_headers,
        ).json() == [{"name": "shipping.md", "path": "shipping.md"}]


def test_platform_admin_tenant_key_lifecycle(app_factory):
    create_app, _, tmp_path = app_factory

    with TestClient(create_app()) as client:
        denied = client.post(
            "/api/v1/admin/tenants",
            headers={"X-API-Key": TENANT_A_ADMIN_KEY},
        )
        assert denied.status_code == 403

        created = client.post(
            "/api/v1/admin/tenants",
            headers={"X-API-Key": PLATFORM_ADMIN_KEY},
        )
        assert created.status_code == 201
        tenant = created.json()

        api_keys_db = tmp_path / "data" / "api_keys.db"
        assert tenant["api_key"].encode() not in api_keys_db.read_bytes()

        headers = {"X-API-Key": tenant["api_key"]}
        assert client.get("/api/v1/knowledge-base", headers=headers).status_code == 200
        key_records = client.get(
            f"/api/v1/admin/tenants/{tenant['tenant_id']}/keys",
            headers={"X-API-Key": PLATFORM_ADMIN_KEY},
        )
        assert key_records.status_code == 200
        assert key_records.json()[0]["key_id"] == tenant["api_key_id"]

        issued = client.post(
            f"/api/v1/admin/tenants/{tenant['tenant_id']}/keys",
            headers={"X-API-Key": PLATFORM_ADMIN_KEY},
            json={"customer_id": "customer-42"},
        )
        assert issued.status_code == 201
        issued_principal = client.app.state.api_keys.authenticate(
            issued.json()["api_key"]
        )
        assert issued_principal.customer_id == "customer-42"

        missing_customer_id = client.post(
            f"/api/v1/admin/tenants/{tenant['tenant_id']}/keys",
            headers={"X-API-Key": PLATFORM_ADMIN_KEY},
            json={},
        )
        assert missing_customer_id.status_code == 422
        blank_customer_id = client.post(
            f"/api/v1/admin/tenants/{tenant['tenant_id']}/keys",
            headers={"X-API-Key": PLATFORM_ADMIN_KEY},
            json={"customer_id": "   "},
        )
        assert blank_customer_id.status_code == 422

        revoked = client.delete(
            f"/api/v1/admin/keys/{tenant['api_key_id']}",
            headers={"X-API-Key": PLATFORM_ADMIN_KEY},
        )
        assert revoked.status_code == 204
        assert client.get("/api/v1/knowledge-base", headers=headers).status_code == 401
        root_key_id = hashlib.sha256(
            PLATFORM_ADMIN_KEY.encode()
        ).hexdigest()[:16]
        assert client.delete(
            f"/api/v1/admin/keys/{root_key_id}",
            headers={"X-API-Key": PLATFORM_ADMIN_KEY},
        ).status_code == 409


def test_rate_limit_is_tenant_scoped(app_factory):
    create_app, _, _ = app_factory

    with TestClient(create_app(rate_limit_requests=1)) as client:
        headers = {"X-API-Key": TENANT_A_CUSTOMER_KEY}
        assert client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "First"},
        ).status_code == 200
        limited = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "Second"},
        )
        assert limited.status_code == 429
        assert "Retry-After" in limited.headers
        assert client.post(
            "/api/v1/chat",
            headers={"X-API-Key": TENANT_B_CUSTOMER_KEY},
            json={"message": "Separate tenant"},
        ).status_code == 200


def test_rate_limit_is_shared_across_app_instances_and_restarts(app_factory):
    create_app, _, _ = app_factory
    headers = {"X-API-Key": TENANT_A_CUSTOMER_KEY}

    with TestClient(create_app(rate_limit_requests=1)) as first_worker:
        assert first_worker.get(
            "/api/v1/conversations",
            headers=headers,
        ).status_code == 200

    with TestClient(create_app(rate_limit_requests=1)) as second_worker:
        limited = second_worker.get(
            "/api/v1/conversations",
            headers=headers,
        )

    assert limited.status_code == 429


def test_rate_limit_backend_failure_fails_closed(app_factory, monkeypatch):
    create_app, _, _ = app_factory

    def backend_failure(tenant_id):
        raise RedisError(
            "redis://user:secret@example.invalid/0 is unavailable"
        )

    with TestClient(create_app()) as client:
        monkeypatch.setattr(
            client.app.state.rate_limiter,
            "allow",
            backend_failure,
        )
        response = client.get(
            "/api/v1/conversations",
            headers={"X-API-Key": TENANT_A_CUSTOMER_KEY},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The support service is unavailable."
    )
    assert "secret" not in response.text


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (TimeoutError("provider timeout"), 504, "The AI service timed out."),
        (
            httpx.TimeoutException("provider timeout"),
            504,
            "The AI service timed out.",
        ),
        (RuntimeError("provider failed"), 503, "The support service is unavailable."),
        (
            ConnectionError("embedding network failure"),
            503,
            "The support service is unavailable.",
        ),
        (
            OpenRouterError(
                "upstream provider details",
                raw_response=httpx.Response(502),
            ),
            502,
            "The AI service is unavailable.",
        ),
    ],
)
def test_api_translates_upstream_errors(
    app_factory,
    error,
    expected_status,
    expected_detail,
):
    create_app, _, _ = app_factory

    def failing_agent(
        message,
        retrieval_service,
        history,
        allowed_tool_names,
        tenant_id=None,
        customer_id=None,
    ):
        raise error

    with TestClient(create_app(agent_runner_override=failing_agent)) as client:
        response = client.post(
            "/api/v1/chat",
            headers={"X-API-Key": TENANT_A_CUSTOMER_KEY},
            json={"message": "Please help"},
        )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail


def test_support_completes_with_mocked_provider_response(
    app_factory,
    monkeypatch,
):
    create_app, _, _ = app_factory
    captured = {}

    def fake_completion(messages, tools):
        captured["messages"] = messages
        captured["tools"] = tools
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Mocked support answer.",
                        tool_calls=[],
                    )
                )
            ]
        )

    monkeypatch.setattr(
        "src.agent.create_chat_completion",
        fake_completion,
    )

    with TestClient(
        create_app(agent_runner_override=run_agent_response)
    ) as client:
        response = client.post(
            "/api/v1/support",
            headers={"X-API-Key": TENANT_A_CUSTOMER_KEY},
            json={"message": "Please help me"},
        )

    assert response.status_code == 200
    assert response.json()["message"] == "Mocked support answer."
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["content"] == "Please help me"
    assert [
        tool["function"]["name"]
        for tool in captured["tools"]
    ] == ["get_order_status"]


def test_support_logs_provider_status_without_provider_message(
    app_factory,
    monkeypatch,
    caplog,
):
    create_app, _, _ = app_factory
    provider_error = OpenRouterError(
        "synthetic private provider detail",
        raw_response=httpx.Response(502),
    )

    def fake_completion(messages, tools):
        raise provider_error

    monkeypatch.setattr(
        "src.agent.create_chat_completion",
        fake_completion,
    )

    with (
        caplog.at_level(logging.WARNING, logger="src.app"),
        TestClient(
            create_app(agent_runner_override=run_agent_response)
        ) as client,
    ):
        response = client.post(
            "/api/v1/support",
            headers={
                "X-API-Key": TENANT_A_CUSTOMER_KEY,
                "X-Request-ID": "mocked-provider-failure",
            },
            json={"message": "Please help me"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "The AI service is unavailable."
    assert "synthetic private provider detail" not in response.text
    assert "synthetic private provider detail" not in caplog.text

    provider_log = next(
        record
        for record in caplog.records
        if record.message.startswith("OpenRouter request failed")
    )
    assert provider_log.message == (
        "OpenRouter request failed "
        "(request_id=mocked-provider-failure, "
        "type=OpenRouterError, upstream_status=502)."
    )
    assert provider_log.request_id == "mocked-provider-failure"
    assert provider_log.error_type == "OpenRouterError"
    assert provider_log.provider_status_code == 502


@pytest.mark.parametrize(
    "filename",
    ["../outside.md", "nested/file.md", "notes.txt"],
)
def test_knowledge_base_rejects_unsafe_document_names(
    app_factory,
    filename,
):
    create_app, _, _ = app_factory

    with TestClient(create_app()) as client:
        response = client.put(
            f"/api/v1/knowledge-base/{filename}",
            headers={"X-API-Key": TENANT_A_ADMIN_KEY},
            json={"content": "# Content"},
        )
        assert response.status_code in {404, 422}
