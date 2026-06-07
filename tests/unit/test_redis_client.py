"""Integration tests for async Redis client wrapper."""

from __future__ import annotations

from typing import Any

import fakeredis.aioredis
import pytest
import redis.asyncio as aioredis

from lib.redis.client import RedisClient


async def test_redis_client_ping_succeeds_with_fakeredis() -> None:
    """Ping returns True against an in-memory fake Redis server."""
    fake = fakeredis.aioredis.FakeRedis()
    client = RedisClient("redis://localhost:6379/0", client=fake)

    assert await client.ping() is True

    await client.close()


async def test_redis_client_connect_and_close() -> None:
    """connect() builds a pool-backed client; close() releases resources."""
    fake = fakeredis.aioredis.FakeRedis()
    client = RedisClient("redis://localhost:6379/0", client=fake)

    assert client.client is fake
    await client.close()
    await client.close()  # idempotent


async def test_redis_client_connect_from_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect() uses redis.asyncio.from_url with pool settings."""
    fake = fakeredis.aioredis.FakeRedis()
    captured: dict[str, Any] = {}

    def mock_from_url(url: str, **kwargs: Any) -> fakeredis.aioredis.FakeRedis:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return fake

    monkeypatch.setattr(aioredis, "from_url", mock_from_url)

    client = await RedisClient.connect("redis://memorystore:6379/0")

    assert captured["url"] == "redis://memorystore:6379/0"
    assert captured["kwargs"]["max_connections"] == 10
    assert captured["kwargs"]["retry_on_timeout"] is True
    assert await client.ping() is True

    await client.close()
