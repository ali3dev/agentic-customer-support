import math
import sqlite3
import time
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from redis import Redis


_REDIS_SLIDING_WINDOW_SCRIPT = """
local redis_time = redis.call('TIME')
local now_ms = tonumber(redis_time[1]) * 1000 + math.floor(
    tonumber(redis_time[2]) / 1000
)
local window_ms = tonumber(ARGV[1]) * 1000
local limit = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now_ms - window_ms)
local count = redis.call('ZCARD', KEYS[1])
if count >= limit then
    local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
    local retry_after = math.max(
        1,
        math.ceil((tonumber(oldest[2]) + window_ms - now_ms) / 1000)
    )
    return {0, retry_after}
end
redis.call('ZADD', KEYS[1], now_ms, ARGV[3])
redis.call('PEXPIRE', KEYS[1], window_ms)
return {1, 0}
"""


class SQLiteSlidingWindowRateLimiter:
    """Persist rate-limit events so workers sharing the data directory agree."""

    def __init__(
        self,
        db_path: str | Path,
        limit: int,
        window_seconds: int,
    ) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("Rate-limit settings must be positive.")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.window_seconds = window_seconds
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_database(self) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rate_limit_events (
                        tenant_id TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rate_limit_tenant_created
                    ON rate_limit_events (tenant_id, created_at)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rate_limit_created
                    ON rate_limit_events (created_at)
                    """
                )
        finally:
            connection.close()

    def allow(self, tenant_id: str) -> tuple[bool, int]:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string.")

        now = time.time()
        cutoff = now - self.window_seconds
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM rate_limit_events WHERE created_at <= ?",
                (cutoff,),
            )
            count, oldest_event = connection.execute(
                """
                SELECT COUNT(*), MIN(created_at)
                FROM rate_limit_events
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()

            if count >= self.limit:
                connection.commit()
                retry_after = max(
                    1,
                    math.ceil(oldest_event + self.window_seconds - now),
                )
                return False, retry_after

            connection.execute(
                """
                INSERT INTO rate_limit_events (tenant_id, created_at)
                VALUES (?, ?)
                """,
                (tenant_id, now),
            )
            connection.commit()
            return True, 0
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


class RedisSlidingWindowRateLimiter:
    """Atomically share rate-limit events across hosts using Redis."""

    def __init__(
        self,
        redis_url: str,
        limit: int,
        window_seconds: int,
    ) -> None:
        if not redis_url.strip():
            raise ValueError("redis_url must be a non-empty string.")
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("Rate-limit settings must be positive.")

        self.limit = limit
        self.window_seconds = window_seconds
        self._client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self._client.ping()
        self._script = self._client.register_script(
            _REDIS_SLIDING_WINDOW_SCRIPT
        )

    def allow(self, tenant_id: str) -> tuple[bool, int]:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string.")

        result = self._script(
            keys=[f"agentic:rate-limit:{sha256(tenant_id.encode()).hexdigest()}"],
            args=[
                self.window_seconds,
                self.limit,
                str(uuid4()),
            ],
        )
        if (
            not isinstance(result, (list, tuple))
            or len(result) != 2
        ):
            raise RuntimeError("Redis returned an invalid rate-limit result.")

        return bool(int(result[0])), int(result[1])

    def close(self) -> None:
        self._client.close()
