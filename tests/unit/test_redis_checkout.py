"""Unit tests for Redis checkout session persistence."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from graphs.checkout_state import initial_checkout_state
from lib.redis.checkout import clear_checkout_session, get_checkout_session, save_checkout_session
from lib.redis.client import RedisClient

_SESSION_ID = "sess-redis-checkout-001"


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.mark.asyncio
async def test_save_and_load_checkout_session(redis_client: RedisClient) -> None:
    state = initial_checkout_state(session_id=_SESSION_ID, currency="LKR")
    state["current_step"] = "delivery_city"
    state["step_valid"] = {"cart": True}
    state["delivery_city"] = "Colombo 03"

    await save_checkout_session(redis_client, _SESSION_ID, state)
    loaded = await get_checkout_session(redis_client, _SESSION_ID)

    assert loaded["current_step"] == "delivery_city"
    assert loaded["delivery_city"] == "Colombo 03"
    assert loaded["step_valid"]["cart"] is True


@pytest.mark.asyncio
async def test_clear_checkout_session_removes_state(redis_client: RedisClient) -> None:
    state = initial_checkout_state(session_id=_SESSION_ID)
    await save_checkout_session(redis_client, _SESSION_ID, state)
    await clear_checkout_session(redis_client, _SESSION_ID)

    assert await get_checkout_session(redis_client, _SESSION_ID) == {}
