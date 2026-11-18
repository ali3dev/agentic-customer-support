import json
import hashlib
import sqlite3

import pytest

from src.security import ApiKeyStore, Principal, load_api_key_config


def test_load_api_key_config_builds_tenant_principals():
    api_key = "x" * 40

    credentials = load_api_key_config(json.dumps({
        api_key: {
            "tenant_id": "tenant-a",
            "role": "tenant_admin",
        },
    }))

    assert credentials == {
        api_key: Principal("tenant-a", "tenant_admin"),
    }


def test_customer_api_key_config_requires_subject_identity():
    api_key = "x" * 40
    credentials = load_api_key_config(json.dumps({
        api_key: {
            "tenant_id": "tenant-a",
            "role": "customer",
            "customer_id": "customer-a",
        },
    }))

    assert credentials[api_key] == Principal(
        "tenant-a",
        "customer",
        "customer-a",
    )


@pytest.mark.parametrize(
    "config",
    [
        "{",
        "[]",
        json.dumps({"short": {"tenant_id": "tenant-a", "role": "customer"}}),
        json.dumps({
            "x" * 40: {
                "tenant_id": "tenant-a",
                "role": ["tenant_admin"],
            },
        }),
        json.dumps({
            "x" * 40: {
                "tenant_id": "tenant-a",
                "role": "root",
            },
        }),
        json.dumps({
            "x" * 40: {
                "tenant_id": "tenant-a",
                "role": "customer",
            },
        }),
    ],
)
def test_load_api_key_config_rejects_malformed_values(config):
    with pytest.raises(ValueError):
        load_api_key_config(config)


def test_api_key_store_hashes_keys_and_rejects_revoked_key(tmp_path):
    api_key = "customer-key-for-test-only-00000000001"
    store = ApiKeyStore(
        tmp_path / "api_keys.db",
        {api_key: Principal("tenant-a", "customer", "customer-a")},
    )

    assert store.authenticate(api_key) == Principal(
        "tenant-a",
        "customer",
        "customer-a",
    )
    assert api_key.encode() not in (tmp_path / "api_keys.db").read_bytes()

    key_id = store.list_keys("tenant-a")[0]["key_id"]
    assert store.revoke_key(key_id, tenant_id="tenant-a")
    assert store.authenticate(api_key) is None


def test_api_key_store_does_not_revoke_last_platform_admin(tmp_path):
    api_key = "platform-key-for-test-only-00000001"
    store = ApiKeyStore(
        tmp_path / "api_keys.db",
        {api_key: Principal("platform", "platform_admin")},
    )

    key_id = store.list_keys("platform")[0]["key_id"]

    with pytest.raises(ValueError, match="last platform administrator"):
        store.revoke_key(key_id)

    assert store.authenticate(api_key) == Principal(
        "platform",
        "platform_admin",
    )


def test_api_key_store_revokes_removed_environment_keys(tmp_path):
    old_platform_key = "old-platform-key-for-test-only-00000001"
    old_customer_key = "old-customer-key-for-test-only-00000001"
    new_platform_key = "new-platform-key-for-test-only-00000001"
    database = tmp_path / "api_keys.db"

    first_store = ApiKeyStore(
        database,
        {
            old_platform_key: Principal("platform", "platform_admin"),
            old_customer_key: Principal(
                "tenant-a",
                "customer",
                "customer-a",
            ),
        },
    )
    assert first_store.authenticate(old_customer_key) is not None

    second_store = ApiKeyStore(
        database,
        {
            new_platform_key: Principal("platform", "platform_admin"),
        },
    )

    assert second_store.authenticate(old_platform_key) is None
    assert second_store.authenticate(old_customer_key) is None
    assert second_store.authenticate(new_platform_key) == Principal(
        "platform",
        "platform_admin",
    )


def test_issued_customer_key_is_bound_to_customer(tmp_path):
    platform_key = "platform-key-for-test-only-00000001"
    store = ApiKeyStore(
        tmp_path / "api_keys.db",
        {platform_key: Principal("platform", "platform_admin")},
    )
    tenant_id = store.create_tenant()["tenant_id"]

    with pytest.raises(ValueError, match="customer_id"):
        store.issue_api_key(tenant_id)

    issued = store.issue_api_key(
        tenant_id,
        customer_id="customer-42",
    )

    assert store.authenticate(issued["api_key"]) == Principal(
        tenant_id,
        "customer",
        "customer-42",
    )


def test_existing_api_key_database_adds_customer_identity_column(tmp_path):
    database = tmp_path / "api_keys.db"
    legacy_key = "legacy-customer-key-for-test-only-0001"
    digest = hashlib.sha256(legacy_key.encode("utf-8")).hexdigest()
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE tenants (
            tenant_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        );
        CREATE TABLE api_keys (
            key_id TEXT PRIMARY KEY,
            key_digest TEXT NOT NULL UNIQUE,
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
            role TEXT NOT NULL,
            revoked_at TEXT,
            source TEXT NOT NULL DEFAULT 'admin',
            created_at TEXT NOT NULL
        );
        INSERT INTO tenants VALUES ('tenant-a', '2026-01-01');
        """
    )
    connection.execute(
        """
        INSERT INTO api_keys (
            key_id, key_digest, tenant_id, role, revoked_at, source, created_at
        ) VALUES (?, ?, 'tenant-a', 'customer', NULL, 'admin', '2026-01-01')
        """,
        (digest[:16], digest),
    )
    connection.commit()
    connection.close()

    store = ApiKeyStore(
        database,
        {
            "platform-key-for-test-only-00000001": Principal(
                "platform",
                "platform_admin",
            ),
        },
    )

    assert store.authenticate(legacy_key) == Principal(
        "tenant-a",
        "customer",
        None,
    )
