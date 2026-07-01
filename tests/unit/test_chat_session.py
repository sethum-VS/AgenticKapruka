"""Tests for chat session rotation and cart clearing."""

from __future__ import annotations

import fakeredis.aioredis
import pytest
from starlette.requests import Request

from lib.chat.session import SESSION_COOKIE_NAME, _sign_thread_id, rotate_chat_thread
from lib.redis.cart import add_item, clear_cart, get_cart
from lib.redis.client import RedisClient


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


def _request_with_cookie(cookie_value: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/chat",
        "headers": [(b"cookie", f"{SESSION_COOKIE_NAME}={cookie_value}".encode())],
    }
    return Request(scope)


def test_rotate_chat_thread_returns_new_signed_cookie() -> None:
    old_thread = "thread-old-abc"
    request = _request_with_cookie(_sign_thread_id(old_thread))
    prior, new_thread, signed = rotate_chat_thread(request)
    assert prior == old_thread
    assert new_thread != old_thread
    assert signed.count(".") == 1


@pytest.mark.asyncio
async def test_clear_cart_on_new_session_leaves_prior_session_empty(
    redis_client: RedisClient,
) -> None:
    old_session = "sess-old"
    new_session = "sess-new"
    await add_item(
        redis_client,
        old_session,
        product_id="cake001",
        name="Chocolate Cake",
        price_amount=4500.0,
        price_currency="LKR",
        quantity=2,
    )
    await clear_cart(redis_client, old_session)
    prior_items = await get_cart(redis_client, old_session)
    new_items = await get_cart(redis_client, new_session)
    assert prior_items == []
    assert new_items == []
