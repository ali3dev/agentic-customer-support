import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


VALID_ROLES = {
    "customer",
    "tenant_admin",
    "platform_admin",
}


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    role: str
    customer_id: str | None = None


def load_api_key_config(raw_config: str | None) -> dict[str, Principal]:
    if not raw_config:
        return {}

    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError("AGENTIC_API_KEYS_JSON must be valid JSON.") from exc

    if not isinstance(config, dict):
        raise ValueError("AGENTIC_API_KEYS_JSON must be a JSON object.")

    credentials = {}

    for api_key, value in config.items():
        if (
            not isinstance(api_key, str)
            or len(api_key) < 32
            or not isinstance(value, dict)
        ):
            raise ValueError(
                "Each API key must be at least 32 characters and map to "
                "a credential object."
            )

        tenant_id = value.get("tenant_id")
        role = value.get("role")
        customer_id = value.get("customer_id")

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Every API key requires a non-empty tenant_id.")

        if not isinstance(role, str) or role not in VALID_ROLES:
            raise ValueError(
                f"API key role must be one of {sorted(VALID_ROLES)}."
            )

        if role == "customer" and (
            not isinstance(customer_id, str)
            or not customer_id.strip()
            or len(customer_id) > 200
        ):
            raise ValueError(
                "Customer API keys require a customer_id of 1 to 200 characters."
            )
        if customer_id is not None and (
            not isinstance(customer_id, str)
            or not customer_id.strip()
            or len(customer_id) > 200
        ):
            raise ValueError(
                "customer_id must contain 1 to 200 characters."
            )
        if role != "customer" and customer_id is not None:
            raise ValueError(
                "Only customer API keys may have a customer_id."
            )

        credentials[api_key] = Principal(
            tenant_id=tenant_id,
            role=role,
            customer_id=customer_id.strip() if customer_id is not None else None,
        )

    return credentials


class ApiKeyStore:
    """Persist only API-key digests and tenant-scoped roles."""

    def __init__(
        self,
        db_path: str | Path,
        initial_credentials: dict[str, Principal] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self._seed_credentials(initial_credentials or {})

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        connection = self._connect()

        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tenants (
                        tenant_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_id TEXT PRIMARY KEY,
                        key_digest TEXT NOT NULL UNIQUE,
                        tenant_id TEXT NOT NULL
                            REFERENCES tenants(tenant_id),
                        role TEXT NOT NULL,
                        customer_id TEXT,
                        revoked_at TEXT,
                        source TEXT NOT NULL DEFAULT 'environment',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(api_keys)"
                    ).fetchall()
                }
                if "source" not in columns:
                    connection.execute(
                        """
                        ALTER TABLE api_keys
                        ADD COLUMN source TEXT NOT NULL DEFAULT 'environment'
                        """
                    )
                if "customer_id" not in columns:
                    connection.execute(
                        """
                        ALTER TABLE api_keys
                        ADD COLUMN customer_id TEXT
                        """
                    )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_api_keys_tenant
                    ON api_keys(tenant_id)
                    """
                )
        finally:
            connection.close()

    def _seed_credentials(
        self,
        credentials: dict[str, Principal],
    ) -> None:
        connection = self._connect()
        now = datetime.now(timezone.utc).isoformat()

        try:
            with connection:
                connection.execute(
                    """
                    UPDATE api_keys
                    SET revoked_at = ?
                    WHERE source = 'environment'
                      AND revoked_at IS NULL
                    """,
                    (now,),
                )

                for api_key, principal in credentials.items():
                    if (
                        not isinstance(api_key, str)
                        or len(api_key) < 32
                        or not isinstance(principal, Principal)
                        or not isinstance(principal.tenant_id, str)
                        or not principal.tenant_id.strip()
                        or not isinstance(principal.role, str)
                        or principal.role not in VALID_ROLES
                        or (
                            principal.role == "customer"
                            and (
                                not isinstance(principal.customer_id, str)
                                or not principal.customer_id.strip()
                                or len(principal.customer_id) > 200
                            )
                        )
                        or (
                            principal.customer_id is not None
                            and (
                                not isinstance(principal.customer_id, str)
                                or not principal.customer_id.strip()
                                or len(principal.customer_id) > 200
                            )
                        )
                        or (
                            principal.role != "customer"
                            and principal.customer_id is not None
                        )
                    ):
                        raise ValueError(
                            "Invalid API key credentials configuration."
                        )

                    digest = hashlib.sha256(
                        api_key.encode("utf-8")
                    ).hexdigest()
                    key_id = digest[:16]

                    connection.execute(
                        """
                        INSERT OR IGNORE INTO tenants (tenant_id, created_at)
                        VALUES (?, ?)
                        """,
                        (principal.tenant_id, now),
                    )
                    connection.execute(
                        """
                        INSERT INTO api_keys (
                            key_id, key_digest, tenant_id, role, customer_id,
                            revoked_at, source, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, NULL, 'environment', ?)
                        ON CONFLICT(key_digest) DO UPDATE SET
                            tenant_id = excluded.tenant_id,
                            role = excluded.role,
                            customer_id = excluded.customer_id,
                            revoked_at = NULL,
                            source = 'environment'
                        """,
                        (
                            key_id,
                            digest,
                            principal.tenant_id,
                            principal.role,
                            (
                                principal.customer_id.strip()
                                if principal.customer_id is not None
                                else None
                            ),
                            now,
                        ),
                    )
        finally:
            connection.close()

    def authenticate(self, api_key: str | None) -> Principal | None:
        if not isinstance(api_key, str) or not api_key:
            return None

        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT tenant_id, role, customer_id
                FROM api_keys
                WHERE key_digest = ? AND revoked_at IS NULL
                """,
                (digest,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            return None

        return Principal(
            tenant_id=row["tenant_id"],
            role=row["role"],
            customer_id=row["customer_id"],
        )

    def create_tenant(self) -> dict[str, str]:
        tenant_id = str(uuid4())
        api_key = secrets.token_urlsafe(32)
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        key_id = digest[:16]
        now = datetime.now(timezone.utc).isoformat()
        connection = self._connect()

        try:
            with connection:
                connection.execute(
                    "INSERT INTO tenants (tenant_id, created_at) VALUES (?, ?)",
                    (tenant_id, now),
                )
                connection.execute(
                    """
                    INSERT INTO api_keys (
                        key_id, key_digest, tenant_id, role, customer_id,
                        revoked_at, source, created_at
                    )
                    VALUES (?, ?, ?, 'tenant_admin', NULL, NULL, 'admin', ?)
                    """,
                    (key_id, digest, tenant_id, now),
                )
        finally:
            connection.close()

        return {
            "tenant_id": tenant_id,
            "api_key_id": key_id,
            "api_key": api_key,
        }

    def issue_api_key(
        self,
        tenant_id: str,
        role: str = "customer",
        customer_id: str | None = None,
    ) -> dict[str, str]:
        if role not in {"customer", "tenant_admin"}:
            raise ValueError("New tenant API keys must use a tenant role.")
        if role == "customer" and (
            not isinstance(customer_id, str)
            or not customer_id.strip()
            or len(customer_id) > 200
        ):
            raise ValueError(
                "Customer API keys require a customer_id of 1 to 200 characters."
            )
        if customer_id is not None and (
            not isinstance(customer_id, str)
            or not customer_id.strip()
            or len(customer_id) > 200
        ):
            raise ValueError(
                "customer_id must contain 1 to 200 characters."
            )
        if role != "customer" and customer_id is not None:
            raise ValueError(
                "Only customer API keys may have a customer_id."
            )

        api_key = secrets.token_urlsafe(32)
        if customer_id is not None:
            customer_id = customer_id.strip()
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        key_id = digest[:16]
        now = datetime.now(timezone.utc).isoformat()
        connection = self._connect()

        try:
            with connection:
                tenant = connection.execute(
                    "SELECT 1 FROM tenants WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()
                if tenant is None:
                    raise LookupError(tenant_id)

                connection.execute(
                    """
                    INSERT INTO api_keys (
                        key_id, key_digest, tenant_id, role, customer_id,
                        revoked_at, source, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, NULL, 'admin', ?)
                    """,
                    (key_id, digest, tenant_id, role, customer_id, now),
                )
        finally:
            connection.close()

        return {
            "api_key_id": key_id,
            "api_key": api_key,
        }

    def list_tenants(self) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            rows = connection.execute(
                """
                SELECT tenant_id, created_at
                FROM tenants
                ORDER BY created_at, tenant_id
                """
            ).fetchall()
        finally:
            connection.close()

        return [
            {
                "tenant_id": row["tenant_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def list_keys(self, tenant_id: str) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            rows = connection.execute(
                """
                SELECT key_id, role, created_at, revoked_at
                FROM api_keys
                WHERE tenant_id = ?
                ORDER BY created_at, key_id
                """,
                (tenant_id,),
            ).fetchall()
        finally:
            connection.close()

        return [dict(row) for row in rows]

    def has_active_platform_admin(self) -> bool:
        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT 1
                FROM api_keys
                WHERE role = 'platform_admin'
                  AND revoked_at IS NULL
                LIMIT 1
                """
            ).fetchone()
        finally:
            connection.close()

        return row is not None

    def revoke_key(self, key_id: str, tenant_id: str | None = None) -> bool:
        connection = self._connect()
        now = datetime.now(timezone.utc).isoformat()

        try:
            with connection:
                target = connection.execute(
                    """
                    SELECT role
                    FROM api_keys
                    WHERE key_id = ? AND revoked_at IS NULL
                    """,
                    (key_id,),
                ).fetchone()
                if target is None:
                    return False

                if target["role"] == "platform_admin":
                    active_platform_admins = connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM api_keys
                        WHERE role = 'platform_admin'
                          AND revoked_at IS NULL
                        """
                    ).fetchone()[0]
                    if active_platform_admins <= 1:
                        raise ValueError(
                            "Cannot revoke the last platform administrator key."
                        )

                if tenant_id is None:
                    cursor = connection.execute(
                        """
                        UPDATE api_keys
                        SET revoked_at = ?
                        WHERE key_id = ? AND revoked_at IS NULL
                        """,
                        (now, key_id),
                    )
                else:
                    cursor = connection.execute(
                        """
                        UPDATE api_keys
                        SET revoked_at = ?
                        WHERE key_id = ?
                          AND tenant_id = ?
                          AND revoked_at IS NULL
                        """,
                        (now, key_id, tenant_id),
                    )
        finally:
            connection.close()

        return cursor.rowcount == 1
