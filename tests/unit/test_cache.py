"""Unit tests for read-only MCP response cache."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from lib.redis.cache import (
    DEFAULT_CACHE_TTL,
    cache_key,
    canonical_args_json,
    get_cached,
    is_cacheable_tool,
    set_cached,
)
from lib.redis.client import RedisClient
from lib.redis.rate_limit import CREATE_ORDER_TOOL

SEARCH_TOOL = "kapruka_search_products"


@pytest.fixture
def redis_client() -> RedisClient:
    """In-memory Redis for cache tests."""
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


def test_canonical_args_json_sorts_keys() -> None:
    """Key order in the input dict must not affect canonical JSON."""
    a = {"q": "birthday cake", "limit": 10, "currency": "LKR"}
    b = {"currency": "LKR", "limit": 10, "q": "birthday cake"}
    assert canonical_args_json(a) == canonical_args_json(b)


def test_cache_key_format_is_tool_colon_sha256() -> None:
    """Cache keys use tool:hash(args) with SHA-256 of canonical JSON."""
    args = {"q": "roses", "limit": 5}
    key = cache_key(SEARCH_TOOL, args)
    assert key.startswith(f"{SEARCH_TOOL}:")
    assert len(key.split(":", 1)[1]) == 64


def test_is_cacheable_tool_blocks_create_order() -> None:
    """Write tool create_order is never cached."""
    assert is_cacheable_tool(SEARCH_TOOL) is True
    assert is_cacheable_tool(CREATE_ORDER_TOOL) is False


async def test_search_products_cache_hit_on_second_identical_call(
    redis_client: RedisClient,
) -> None:
    """Second identical search_products call returns cached response."""
    args = {"q": "birthday cake", "limit": 10, "response_format": "json"}
    response = '{"results": [{"id": "cake001", "name": "Chocolate Cake"}]}'

    assert await get_cached(redis_client, SEARCH_TOOL, args) is None

    await set_cached(redis_client, SEARCH_TOOL, args, response)

    cached = await get_cached(redis_client, SEARCH_TOOL, args)
    assert cached == response

    key = cache_key(SEARCH_TOOL, args)
    ttl = await redis_client.client.ttl(key)
    assert 0 < ttl <= DEFAULT_CACHE_TTL


async def test_reordered_args_produce_cache_hit(redis_client: RedisClient) -> None:
    """Sorted-key canonicalization yields a hit for equivalent arg dicts."""
    first = {"q": "tea gift", "limit": 10}
    second = {"limit": 10, "q": "tea gift"}
    response = '{"results": []}'

    await set_cached(redis_client, SEARCH_TOOL, first, response)
    assert await get_cached(redis_client, SEARCH_TOOL, second) == response


async def test_create_order_bypasses_cache(redis_client: RedisClient) -> None:
    """create_order never reads from or writes to the cache."""
    args = {
        "cart": [{"product_id": "cake001", "quantity": 1}],
        "recipient": {"name": "Ada", "phone": "0771234567"},
        "delivery": {
            "address": "1 Main St",
            "city": "Colombo 03",
            "date": "2026-06-10",
        },
        "sender": {"name": "Bob", "anonymous": False},
    }
    response = '{"checkout_url": "https://kapruka.com/pay/abc"}'

    await set_cached(redis_client, CREATE_ORDER_TOOL, args, response)
    assert await get_cached(redis_client, CREATE_ORDER_TOOL, args) is None

    keys = [key async for key in redis_client.client.scan_iter(match=f"{CREATE_ORDER_TOOL}:*")]
    assert keys == []
