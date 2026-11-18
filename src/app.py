import logging
import math
import os
import re
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Callable

import httpx
from requests.exceptions import RequestException
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from openrouter.errors import (
    EdgeNetworkTimeoutResponseError,
    GatewayTimeoutResponseError,
    OpenRouterError,
    ProviderOverloadedResponseError,
    RequestTimeoutResponseError,
    ServiceUnavailableResponseError,
    TooManyRequestsResponseError,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator
from redis.exceptions import RedisError
from starlette.concurrency import run_in_threadpool

from src.agent import run_agent_response
from src.conversations import ConversationNotFound, ConversationStore
from src.rag.embeddings import EmbeddingService
from src.rate_limiter import (
    RedisSlidingWindowRateLimiter,
    SQLiteSlidingWindowRateLimiter,
)
from src.schemas.support import AgentResponse, SourceMetadata, TokenUsage
from src.security import ApiKeyStore, Principal, load_api_key_config
from src.tenant_services import TenantServiceRegistry

logger = logging.getLogger(__name__)

ROLE_TOOL_PERMISSIONS = {
    "customer": {"get_order_status"},
    "tenant_admin": {"get_order_status"},
    "platform_admin": {"get_order_status"},
}


@dataclass(frozen=True)
class AppSettings:
    initial_api_keys: dict[str, Principal] = field(default_factory=dict)
    data_dir: Path = Path("data")
    knowledge_base_dir: Path = Path("data/tenant_knowledge_bases")
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    rate_limit_redis_url: str | None = None
    relevance_threshold: float = 0.6

    @classmethod
    def from_environment(cls) -> "AppSettings":
        initial_api_keys = load_api_key_config(
            os.getenv("AGENTIC_API_KEYS_JSON")
        )
        return cls(
            initial_api_keys=initial_api_keys,
            data_dir=Path(os.getenv("AGENTIC_DATA_DIR", "data")),
            knowledge_base_dir=Path(
                os.getenv(
                    "AGENTIC_KNOWLEDGE_BASE_DIR",
                    "data/tenant_knowledge_bases",
                )
            ),
            rate_limit_requests=int(
                os.getenv("AGENTIC_RATE_LIMIT_REQUESTS", "60")
            ),
            rate_limit_window_seconds=int(
                os.getenv("AGENTIC_RATE_LIMIT_WINDOW_SECONDS", "60")
            ),
            rate_limit_redis_url=os.getenv(
                "AGENTIC_RATE_LIMIT_REDIS_URL"
            ) or None,
            relevance_threshold=float(
                os.getenv("AGENTIC_RELEVANCE_THRESHOLD", "0.6")
            ),
        )

    def __post_init__(self) -> None:
        if (
            isinstance(self.rate_limit_requests, bool)
            or not isinstance(self.rate_limit_requests, int)
            or self.rate_limit_requests <= 0
        ):
            raise ValueError("rate_limit_requests must be positive.")
        if (
            isinstance(self.rate_limit_window_seconds, bool)
            or not isinstance(self.rate_limit_window_seconds, int)
            or self.rate_limit_window_seconds <= 0
        ):
            raise ValueError("rate_limit_window_seconds must be positive.")
        if (
            isinstance(self.relevance_threshold, bool)
            or not isinstance(self.relevance_threshold, (int, float))
            or not math.isfinite(self.relevance_threshold)
            or not -1.0 <= self.relevance_threshold <= 1.0
        ):
            raise ValueError(
                "relevance_threshold must be between -1 and 1."
            )
        if self.rate_limit_redis_url is not None and not (
            isinstance(self.rate_limit_redis_url, str)
            and self.rate_limit_redis_url.strip()
        ):
            raise ValueError(
                "rate_limit_redis_url must be a non-empty Redis URL."
            )


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    sources: list[SourceMetadata]
    usage: TokenUsage | None = None
    conversation_id: uuid.UUID


class KnowledgeDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=1_000_000)


class CustomerApiKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=200)

    @field_validator("customer_id")
    @classmethod
    def normalize_customer_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("customer_id must not be blank.")
        return normalized


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Customer Support Dashboard</title>
  <style>
    body { font: 16px system-ui, sans-serif; margin: 2rem auto; max-width: 760px; padding: 0 1rem; }
    label { display: block; margin: 1rem 0 .35rem; }
    input, textarea, button { box-sizing: border-box; font: inherit; padding: .6rem; width: 100%; }
    textarea { min-height: 7rem; }
    button { cursor: pointer; margin-top: .75rem; }
    pre { background: #f3f4f6; overflow-wrap: anywhere; padding: 1rem; white-space: pre-wrap; }
    .note { color: #555; }
    .row { display: flex; gap: .75rem; }
    .row button { flex: 1; }
    .item { background: #f3f4f6; border: 0; margin: .25rem 0; text-align: left; }
  </style>
</head>
<body>
  <h1>Customer Support Dashboard</h1>
  <p class="note">Prototype dashboard. The API key remains in this page's memory only.</p>
  <label for="key">Tenant API key</label>
  <input id="key" type="password" autocomplete="off">
  <label for="message">Customer message</label>
  <textarea id="message">What is your return policy?</textarea>
  <div class="row">
    <button id="send">Send support request</button>
    <button id="newConversation">New conversation</button>
  </div>
  <h2>Response</h2>
  <pre id="result">No request sent.</pre>
  <h2>Recent conversations</h2>
  <button id="refresh">Refresh conversations</button>
  <div id="conversations">No conversations loaded.</div>
  <h2>Usage (tenant admin)</h2>
  <button id="refreshUsage">Refresh token usage</button>
  <pre id="usage">No usage loaded.</pre>
  <h2>Knowledge base (tenant admin)</h2>
  <label for="documentName">Markdown filename</label>
  <input id="documentName" placeholder="returns.md">
  <label for="documentContent">Markdown content</label>
  <textarea id="documentContent" placeholder="# Returns"></textarea>
  <div class="row">
    <button id="saveDocument">Save document</button>
    <button id="refreshKnowledge">Refresh documents</button>
  </div>
  <div id="knowledgeBase">No documents loaded.</div>
  <h2>Tenant administration (platform admin)</h2>
  <button id="createTenant">Create tenant and admin key</button>
  <pre id="tenantResult">No tenant created.</pre>
  <script>
    const keyInput = document.getElementById("key");
    const result = document.getElementById("result");
    const conversations = document.getElementById("conversations");
    const knowledgeBase = document.getElementById("knowledgeBase");
    let conversationId = null;
    async function request(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": keyInput.value,
          ...(options.headers || {})
        }
      });
      const payload = response.status === 204 ? {} : await response.json();
      if (!response.ok) throw new Error(payload.detail || "Request failed.");
      return payload;
    }
    document.getElementById("newConversation").addEventListener("click", () => {
      conversationId = null;
      result.textContent = "Started a new conversation.";
    });
    document.getElementById("send").addEventListener("click", async () => {
      result.textContent = "Working…";
      try {
        const payload = await request("/api/v1/support", {
          method: "POST",
          body: JSON.stringify({
            message: document.getElementById("message").value,
            conversation_id: conversationId
          })
        });
        conversationId = payload.conversation_id;
        result.textContent = JSON.stringify(payload, null, 2);
      } catch (error) {
        result.textContent = error.message;
      }
    });
    document.getElementById("refresh").addEventListener("click", async () => {
      conversations.textContent = "Loading…";
      try {
        const payload = await request("/api/v1/conversations");
        conversations.replaceChildren();
        if (!payload.length) conversations.textContent = "No conversations.";
        for (const item of payload) {
          const row = document.createElement("div");
          const openButton = document.createElement("button");
          openButton.className = "item";
          openButton.textContent = `${item.conversation_id} (${item.message_count} messages)`;
          openButton.addEventListener("click", async () => {
            try {
              const details = await request(
                `/api/v1/conversations/${item.conversation_id}`
              );
              conversationId = item.conversation_id;
              result.textContent = JSON.stringify(details, null, 2);
            } catch (error) {
              result.textContent = error.message;
            }
          });
          const deleteButton = document.createElement("button");
          deleteButton.className = "item";
          deleteButton.textContent = "Delete conversation";
          deleteButton.addEventListener("click", async () => {
            if (!window.confirm("Delete this conversation?")) return;
            try {
              await request(`/api/v1/conversations/${item.conversation_id}`, {
                method: "DELETE"
              });
              if (conversationId === item.conversation_id) conversationId = null;
              document.getElementById("refresh").click();
            } catch (error) {
              result.textContent = error.message;
            }
          });
          row.append(openButton, deleteButton);
          conversations.append(row);
        }
      } catch (error) {
        conversations.textContent = error.message;
      }
    });
    document.getElementById("refreshUsage").addEventListener("click", async () => {
      const usage = document.getElementById("usage");
      usage.textContent = "Loading…";
      try {
        usage.textContent = JSON.stringify(
          await request("/api/v1/admin/usage"),
          null,
          2
        );
      } catch (error) {
        usage.textContent = error.message;
      }
    });
    document.getElementById("refreshKnowledge").addEventListener("click", async () => {
      knowledgeBase.textContent = "Loading…";
      try {
        const payload = await request("/api/v1/knowledge-base");
        knowledgeBase.replaceChildren();
        if (!payload.length) knowledgeBase.textContent = "No documents.";
        for (const item of payload) {
          const button = document.createElement("button");
          button.className = "item";
          button.textContent = `Delete ${item.path}`;
          button.addEventListener("click", async () => {
            if (!window.confirm(`Delete ${item.path}?`)) return;
            try {
              await request(`/api/v1/knowledge-base/${encodeURIComponent(item.name)}`, {
                method: "DELETE"
              });
              document.getElementById("refreshKnowledge").click();
            } catch (error) {
              knowledgeBase.textContent = error.message;
            }
          });
          knowledgeBase.append(button);
        }
      } catch (error) {
        knowledgeBase.textContent = error.message;
      }
    });
    document.getElementById("saveDocument").addEventListener("click", async () => {
      knowledgeBase.textContent = "Saving and indexing…";
      try {
        const name = document.getElementById("documentName").value;
        const content = document.getElementById("documentContent").value;
        await request(`/api/v1/knowledge-base/${encodeURIComponent(name)}`, {
          method: "PUT",
          body: JSON.stringify({content})
        });
        document.getElementById("refreshKnowledge").click();
      } catch (error) {
        knowledgeBase.textContent = error.message;
      }
    });
    document.getElementById("createTenant").addEventListener("click", async () => {
      const tenantResult = document.getElementById("tenantResult");
      tenantResult.textContent = "Creating…";
      try {
        const payload = await request("/api/v1/admin/tenants", {method: "POST"});
        tenantResult.textContent = JSON.stringify(payload, null, 2);
      } catch (error) {
        tenantResult.textContent = error.message;
      }
    });
  </script>
</body>
</html>"""


def create_app(
    settings: AppSettings | None = None,
    embedding_service_factory: Callable[[], EmbeddingService] = EmbeddingService,
    agent_runner: Callable[..., AgentResponse] = run_agent_response,
) -> FastAPI:
    app_settings = settings or AppSettings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.api_keys = ApiKeyStore(
            app_settings.data_dir / "api_keys.db",
            initial_credentials=app_settings.initial_api_keys,
        )
        if not app.state.api_keys.has_active_platform_admin():
            raise RuntimeError(
                "Configure at least one platform_admin API key before startup."
            )
        app.state.conversations = ConversationStore(
            app_settings.data_dir / "conversations.db"
        )
        app.state.tenant_services = TenantServiceRegistry(
            embedding_service=embedding_service_factory(),
            data_dir=app_settings.data_dir,
            knowledge_base_dir=app_settings.knowledge_base_dir,
            relevance_threshold=app_settings.relevance_threshold,
        )
        if app_settings.rate_limit_redis_url is not None:
            app.state.rate_limiter = RedisSlidingWindowRateLimiter(
                app_settings.rate_limit_redis_url,
                app_settings.rate_limit_requests,
                app_settings.rate_limit_window_seconds,
            )
        else:
            app.state.rate_limiter = SQLiteSlidingWindowRateLimiter(
                app_settings.data_dir / "rate_limits.db",
                app_settings.rate_limit_requests,
                app_settings.rate_limit_window_seconds,
            )
        try:
            yield
        finally:
            close = getattr(app.state.rate_limiter, "close", None)
            if close is not None:
                close()

    app = FastAPI(
        title="Agentic Customer Support API",
        version="0.2.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("X-Request-ID")
        if not request_id or not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", request_id):
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "tenant_id": getattr(
                    getattr(request.state, "principal", None),
                    "tenant_id",
                    None,
                ),
            },
        )
        return response

    def enforce_rate_limit(principal: Principal) -> None:
        allowed, retry_after = app.state.rate_limiter.allow(
            principal.tenant_id
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded.",
                headers={"Retry-After": str(retry_after)},
            )

    def authenticated_principal(
        request: Request,
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> Principal:
        principal = request.app.state.api_keys.authenticate(x_api_key)
        if principal is None:
            raise HTTPException(
                status_code=401,
                detail="A valid X-API-Key is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        request.state.principal = principal
        enforce_rate_limit(principal)
        return principal

    def require_tenant_admin(
        principal: Annotated[Principal, Depends(authenticated_principal)],
    ) -> Principal:
        if principal.role not in {"tenant_admin", "platform_admin"}:
            raise HTTPException(
                status_code=403,
                detail="Tenant administrator permission is required.",
            )
        return principal

    def require_platform_admin(
        principal: Annotated[Principal, Depends(authenticated_principal)],
    ) -> Principal:
        if principal.role != "platform_admin":
            raise HTTPException(
                status_code=403,
                detail="Platform administrator permission is required.",
            )
        return principal

    def conversation_customer_scope(
        principal: Principal,
    ) -> str | None:
        if principal.role != "customer":
            return None
        if (
            not isinstance(principal.customer_id, str)
            or not principal.customer_id.strip()
        ):
            raise HTTPException(
                status_code=403,
                detail="Customer identity is required for conversations.",
            )
        return principal.customer_id

    async def execute_chat(
        payload: ChatRequest,
        principal: Principal,
    ) -> ChatResponse:
        customer_id = conversation_customer_scope(principal)
        conversation_id = str(
            payload.conversation_id or uuid.uuid4()
        )
        history = []
        if payload.conversation_id is not None:
            history = app.state.conversations.get_history(
                principal.tenant_id,
                conversation_id,
                limit=20,
                customer_id=customer_id,
            )

        retrieval_service = (
            app.state.tenant_services.get_retrieval_service(
                principal.tenant_id
            )
        )
        agent_response = await run_in_threadpool(
            agent_runner,
            payload.message,
            retrieval_service,
            history,
            (
                ROLE_TOOL_PERMISSIONS[principal.role]
                if (
                    principal.role == "customer"
                    and principal.customer_id is not None
                )
                else set()
            ),
            tenant_id=principal.tenant_id,
            customer_id=principal.customer_id,
        )
        sources = [
            source.model_dump(mode="json")
            for source in agent_response.sources
        ]
        usage = (
            agent_response.usage.model_dump(mode="json")
            if agent_response.usage is not None
            else None
        )
        app.state.conversations.save_turn(
            tenant_id=principal.tenant_id,
            conversation_id=conversation_id,
            user_message=payload.message,
            assistant_message=agent_response.message,
            sources=sources,
            usage=usage,
            customer_id=customer_id,
        )

        return ChatResponse(
            message=agent_response.message,
            sources=agent_response.sources,
            usage=agent_response.usage,
            conversation_id=uuid.UUID(conversation_id),
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(_dashboard_html())

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        principal: Annotated[Principal, Depends(authenticated_principal)],
    ) -> ChatResponse:
        return await execute_chat(payload, principal)

    @app.post("/api/v1/support", response_model=ChatResponse)
    async def support(
        payload: ChatRequest,
        principal: Annotated[Principal, Depends(authenticated_principal)],
    ) -> ChatResponse:
        return await execute_chat(payload, principal)

    @app.get("/api/v1/conversations")
    async def list_conversations(
        principal: Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[dict]:
        return app.state.conversations.list_conversations(
            principal.tenant_id,
            customer_id=conversation_customer_scope(principal),
        )

    @app.get("/api/v1/admin/usage")
    async def tenant_usage(
        principal: Annotated[Principal, Depends(require_tenant_admin)],
    ) -> dict:
        return app.state.conversations.usage_summary(
            principal.tenant_id
        )

    @app.get("/api/v1/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: uuid.UUID,
        principal: Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict:
        messages = app.state.conversations.get_messages(
            principal.tenant_id,
            str(conversation_id),
            customer_id=conversation_customer_scope(principal),
        )
        return {
            "conversation_id": conversation_id,
            "messages": messages,
        }

    @app.delete("/api/v1/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(
        conversation_id: uuid.UUID,
        principal: Annotated[Principal, Depends(authenticated_principal)],
    ) -> Response:
        deleted = app.state.conversations.delete_conversation(
            principal.tenant_id,
            str(conversation_id),
            customer_id=conversation_customer_scope(principal),
        )
        if not deleted:
            raise ConversationNotFound(str(conversation_id))
        return Response(status_code=204)

    @app.get("/api/v1/knowledge-base")
    async def list_knowledge_base(
        principal: Annotated[Principal, Depends(require_tenant_admin)],
    ) -> list[dict[str, str]]:
        root = app.state.tenant_services.knowledge_base_path(
            principal.tenant_id
        )
        if not root.exists():
            return []

        return [
            {
                "name": path.name,
                "path": path.relative_to(root).as_posix(),
            }
            for path in sorted(root.rglob("*.md"))
            if path.is_file()
        ]

    @app.put("/api/v1/knowledge-base/{document_name}")
    async def put_knowledge_document(
        document_name: str,
        payload: KnowledgeDocumentRequest,
        principal: Annotated[Principal, Depends(require_tenant_admin)],
    ) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\.md", document_name):
            raise HTTPException(
                status_code=422,
                detail="document_name must be a simple Markdown filename.",
            )

        try:
            result = await run_in_threadpool(
                app.state.tenant_services.put_knowledge_document,
                principal.tenant_id,
                document_name,
                payload.content,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            sqlite3.Error,
            RequestException,
        ):
            logger.exception(
                "Tenant knowledge-base synchronization failed.",
                extra={"tenant_id": principal.tenant_id},
            )
            raise HTTPException(
                status_code=503,
                detail="Knowledge base could not be synchronized.",
            ) from None

        return {
            "document": document_name,
            "ingestion": result,
        }

    @app.delete("/api/v1/knowledge-base/{document_name}", status_code=200)
    async def delete_knowledge_document(
        document_name: str,
        principal: Annotated[Principal, Depends(require_tenant_admin)],
    ) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\.md", document_name):
            raise HTTPException(
                status_code=422,
                detail="document_name must be a simple Markdown filename.",
            )

        try:
            result = await run_in_threadpool(
                app.state.tenant_services.delete_knowledge_document,
                principal.tenant_id,
                document_name,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Knowledge-base document not found.",
            ) from None
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            sqlite3.Error,
            RequestException,
        ):
            logger.exception(
                "Tenant knowledge-base synchronization failed.",
                extra={"tenant_id": principal.tenant_id},
            )
            raise HTTPException(
                status_code=503,
                detail="Knowledge base could not be synchronized.",
            ) from None

        return {
            "deleted": document_name,
            "ingestion": result,
        }

    @app.get("/api/v1/admin/tenants")
    async def list_tenants(
        _: Annotated[Principal, Depends(require_platform_admin)],
    ) -> list[dict]:
        return app.state.api_keys.list_tenants()

    @app.post("/api/v1/admin/tenants", status_code=201)
    async def create_tenant(
        _: Annotated[Principal, Depends(require_platform_admin)],
    ) -> dict[str, str]:
        return app.state.api_keys.create_tenant()

    @app.get("/api/v1/admin/tenants/{tenant_id}/keys")
    async def list_tenant_keys(
        tenant_id: str,
        _: Annotated[Principal, Depends(require_platform_admin)],
    ) -> list[dict]:
        return app.state.api_keys.list_keys(tenant_id)

    @app.post("/api/v1/admin/tenants/{tenant_id}/keys", status_code=201)
    async def create_tenant_key(
        tenant_id: str,
        payload: CustomerApiKeyRequest,
        _: Annotated[Principal, Depends(require_platform_admin)],
    ) -> dict[str, str]:
        try:
            return app.state.api_keys.issue_api_key(
                tenant_id,
                role="customer",
                customer_id=payload.customer_id,
            )
        except LookupError:
            raise HTTPException(
                status_code=404,
                detail="Tenant not found.",
            ) from None

    @app.get("/api/v1/admin/tenants/{tenant_id}/usage")
    async def platform_tenant_usage(
        tenant_id: str,
        _: Annotated[Principal, Depends(require_platform_admin)],
    ) -> dict:
        return app.state.conversations.usage_summary(tenant_id)

    @app.delete("/api/v1/admin/keys/{key_id}", status_code=204)
    async def revoke_api_key(
        key_id: str,
        _: Annotated[Principal, Depends(require_platform_admin)],
    ) -> Response:
        try:
            revoked = app.state.api_keys.revoke_key(key_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from None

        if not revoked:
            raise HTTPException(status_code=404, detail="API key not found.")
        return Response(status_code=204)

    @app.exception_handler(ConversationNotFound)
    async def handle_missing_conversation(
        _request: Request,
        _exc: ConversationNotFound,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": "Conversation not found."},
        )

    @app.exception_handler(httpx.TimeoutException)
    async def handle_upstream_timeout(
        request: Request,
        _exc: httpx.TimeoutException,
    ) -> JSONResponse:
        logger.warning(
            "Upstream request timed out.",
            extra={"request_id": request.state.request_id},
        )
        return JSONResponse(
            status_code=504,
            content={"detail": "The AI service timed out."},
        )

    @app.exception_handler(TimeoutError)
    async def handle_python_timeout(
        request: Request,
        _exc: TimeoutError,
    ) -> JSONResponse:
        logger.warning(
            "Upstream request timed out.",
            extra={"request_id": request.state.request_id},
        )
        return JSONResponse(
            status_code=504,
            content={"detail": "The AI service timed out."},
        )

    @app.exception_handler(httpx.HTTPError)
    async def handle_upstream_http_error(
        request: Request,
        _exc: httpx.HTTPError,
    ) -> JSONResponse:
        logger.exception(
            "Upstream HTTP request failed.",
            extra={"request_id": request.state.request_id},
        )
        return JSONResponse(
            status_code=502,
            content={"detail": "The AI service is unavailable."},
        )

    @app.exception_handler(RuntimeError)
    async def handle_agent_runtime_error(
        request: Request,
        _exc: RuntimeError,
    ) -> JSONResponse:
        logger.exception(
            "Agent request failed.",
            extra={"request_id": request.state.request_id},
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "The support service is unavailable."},
        )

    async def handle_openrouter_timeout(
        request: Request,
        exc: OpenRouterError,
    ) -> JSONResponse:
        logger.warning(
            "OpenRouter request timed out.",
            extra={
                "request_id": request.state.request_id,
                "error_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=504,
            content={"detail": "The AI service timed out."},
        )

    async def handle_openrouter_rate_limit(
        request: Request,
        exc: OpenRouterError,
    ) -> JSONResponse:
        logger.warning(
            "OpenRouter rate limit reached.",
            extra={
                "request_id": request.state.request_id,
                "error_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "5"},
            content={"detail": "The AI service is temporarily busy."},
        )

    async def handle_openrouter_unavailable(
        request: Request,
        exc: OpenRouterError,
    ) -> JSONResponse:
        logger.warning(
            "OpenRouter service unavailable.",
            extra={
                "request_id": request.state.request_id,
                "error_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "The AI service is temporarily unavailable."},
        )

    async def handle_openrouter_error(
        request: Request,
        exc: OpenRouterError,
    ) -> JSONResponse:
        logger.warning(
            (
                "OpenRouter request failed "
                "(request_id=%s, type=%s, upstream_status=%s)."
            ),
            request.state.request_id,
            type(exc).__name__,
            exc.status_code,
            extra={
                "request_id": request.state.request_id,
                "error_type": type(exc).__name__,
                "provider_status_code": exc.status_code,
            },
        )
        return JSONResponse(
            status_code=502,
            content={"detail": "The AI service is unavailable."},
        )

    @app.exception_handler(RequestException)
    async def handle_embedding_network_error(
        request: Request,
        exc: RequestException,
    ) -> JSONResponse:
        logger.warning(
            "Embedding service network request failed.",
            extra={
                "request_id": request.state.request_id,
                "error_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "The support service is unavailable."},
        )

    for timeout_error in (
        RequestTimeoutResponseError,
        GatewayTimeoutResponseError,
        EdgeNetworkTimeoutResponseError,
    ):
        app.add_exception_handler(
            timeout_error,
            handle_openrouter_timeout,
        )

    app.add_exception_handler(
        TooManyRequestsResponseError,
        handle_openrouter_rate_limit,
    )
    for unavailable_error in (
        ProviderOverloadedResponseError,
        ServiceUnavailableResponseError,
    ):
        app.add_exception_handler(
            unavailable_error,
            handle_openrouter_unavailable,
        )
    app.add_exception_handler(
        OpenRouterError,
        handle_openrouter_error,
    )

    @app.exception_handler(sqlite3.Error)
    async def handle_database_error(
        request: Request,
        _exc: sqlite3.Error,
    ) -> JSONResponse:
        logger.exception(
            "Application database request failed.",
            extra={"request_id": request.state.request_id},
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "The support service is unavailable."},
        )

    @app.exception_handler(RedisError)
    async def handle_rate_limit_backend_error(
        request: Request,
        exc: RedisError,
    ) -> JSONResponse:
        logger.error(
            "Rate-limit backend request failed.",
            extra={
                "request_id": request.state.request_id,
                "error_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "The support service is unavailable."},
        )

    return app


app = create_app()
