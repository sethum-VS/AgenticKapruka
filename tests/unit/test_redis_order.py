"""Unit tests for pending order metadata in Redis."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from lib.redis.client import RedisClient
from lib.redis.order import get_pending_order, store_pending_order

_SESSION_ID = "sess-redis-order-001"


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.mark.asyncio
async def test_store_and_get_pending_order(redis_client: RedisClient) -> None:
    await store_pending_order(
        redis_client,
        _SESSION_ID,
        order_ref="ORD-20260608-7823",
        expires_at="2026-06-08T12:30:00+05:30",
    )

    pending = await get_pending_order(redis_client, _SESSION_ID)
    assert pending is not None
    assert pending.order_ref == "ORD-20260608-7823"
    assert pending.expires_at == "2026-06-08T12:30:00+05:30"


@pytest.mark.asyncio
async def test_get_pending_order_returns_none_when_missing(
    redis_client: RedisClient,
) -> None:
    assert await get_pending_order(redis_client, "unknown-session") is None
