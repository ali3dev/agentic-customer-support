from concurrent.futures import ThreadPoolExecutor

from src.rate_limiter import (
    RedisSlidingWindowRateLimiter,
    SQLiteSlidingWindowRateLimiter,
)


def test_rate_limit_is_shared_by_instances_and_survives_restart(tmp_path):
    database = tmp_path / "rate_limits.db"
    first_worker = SQLiteSlidingWindowRateLimiter(database, 1, 60)
    second_worker = SQLiteSlidingWindowRateLimiter(database, 1, 60)

    assert first_worker.allow("tenant-a") == (True, 0)
    assert second_worker.allow("tenant-a")[0] is False
    assert second_worker.allow("tenant-b") == (True, 0)

    restarted_worker = SQLiteSlidingWindowRateLimiter(database, 1, 60)
    assert restarted_worker.allow("tenant-a")[0] is False


def test_concurrent_workers_cannot_exceed_limit(tmp_path):
    limiter = SQLiteSlidingWindowRateLimiter(
        tmp_path / "rate_limits.db",
        limit=3,
        window_seconds=60,
    )

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(
            limiter.allow,
            ["tenant-a"] * 12,
        ))

    assert sum(allowed for allowed, _ in results) == 3


def test_redis_rate_limiter_uses_atomic_script_and_hashed_tenant_key(
    monkeypatch,
):
    class FakeRedis:
        def __init__(self):
            self.pinged = False
            self.script_text = ""
            self.calls = []
            self.responses = iter(([1, 0], [0, 4]))

        def ping(self):
            self.pinged = True

        def register_script(self, script):
            self.script_text = script

            def execute(*, keys, args):
                self.calls.append((keys, args))
                return next(self.responses)

            return execute

        def close(self):
            pass

    client = FakeRedis()
    monkeypatch.setattr(
        "src.rate_limiter.Redis.from_url",
        lambda *args, **kwargs: client,
    )
    limiter = RedisSlidingWindowRateLimiter(
        "redis://localhost:6379/0",
        limit=1,
        window_seconds=60,
    )

    assert limiter.allow("tenant-a") == (True, 0)
    assert limiter.allow("tenant-a") == (False, 4)
    assert client.pinged
    assert "ZREMRANGEBYSCORE" in client.script_text
    assert client.calls[0][0] == client.calls[1][0]
    assert "tenant-a" not in client.calls[0][0][0]
    limiter.close()
