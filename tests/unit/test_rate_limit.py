"""Unit tests for sliding-window Redis rate limiter."""

from __future__ import annotations

import fakeredis.aioredis
import pytest
from fastapi import FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient

from lib.redis.client import RedisClient
from lib.redis.rate_limit import (
    CREATE_ORDER_MAX_REQUESTS,
    CREATE_ORDER_TOOL,
    GLOBAL_MAX_REQUESTS,
    RateLimitExceeded,
    check_rate_limit,
    rate_limit_key,
    retry_after_header,
)


@pytest.fixture
def redis_client() -> RedisClient:
    """In-memory Redis for rate-limit tests."""
    fake = fakeredis.aioredis.FakeRedis()
    return RedisClient("redis://localhost:6379/0", client=fake)


async def test_global_limit_allows_sixtieth_request(redis_client: RedisClient) -> None:
    """The 60th request within one minute is still allowed."""
    ip = "203.0.113.10"
    for _ in range(GLOBAL_MAX_REQUESTS):
        await check_rate_limit(redis_client, ip, "kapruka_search_products")

    key = rate_limit_key(ip, "global")
    count = await redis_client.client.zcard(key)
    assert count == GLOBAL_MAX_REQUESTS


async def test_global_limit_blocks_sixty_first_request(redis_client: RedisClient) -> None:
    """The 61st global request raises RateLimitExceeded with retry_after."""
    ip = "203.0.113.11"
    for _ in range(GLOBAL_MAX_REQUESTS):
        await check_rate_limit(redis_client, ip, "kapruka_get_product")

    with pytest.raises(RateLimitExceeded) as exc_info:
        await check_rate_limit(redis_client, ip, "kapruka_list_categories")

    exc = exc_info.value
    assert exc.limit_type == "global"
    assert exc.retry_after_seconds >= 1
    assert exc.ip == ip


async def test_sixty_first_request_returns_429_with_retry_after_header(
    redis_client: RedisClient,
) -> None:
    """Simulate 61st global request through HTTP and verify 429 + Retry-After."""
    ip = "198.51.100.7"

    app = FastAPI()

    @app.get("/mcp-proxy")
    async def mcp_proxy(request: Request) -> dict[str, bool]:
        default_ip = request.client.host if request.client else ip
        client_ip = request.headers.get("x-forwarded-for", default_ip)
        try:
            await check_rate_limit(redis_client, client_ip, "kapruka_search_products")
        except RateLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail="Too Many Requests",
                headers=retry_after_header(exc),
            ) from exc
        return {"ok": True}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(GLOBAL_MAX_REQUESTS):
            response = await client.get("/mcp-proxy", headers={"x-forwarded-for": ip})
            assert response.status_code == 200

        blocked = await client.get("/mcp-proxy", headers={"x-forwarded-for": ip})
        assert blocked.status_code == 429
        assert "retry-after" in {name.lower() for name in blocked.headers}
        retry_after = int(blocked.headers["retry-after"])
        assert retry_after >= 1


async def test_create_order_uses_separate_redis_key(redis_client: RedisClient) -> None:
    """create_order increments both global and create_order sorted-set keys."""
    ip = "203.0.113.12"
    await check_rate_limit(redis_client, ip, CREATE_ORDER_TOOL)

    global_key = rate_limit_key(ip, "global")
    order_key = rate_limit_key(ip, "create_order")
    assert await redis_client.client.zcard(global_key) == 1
    assert await redis_client.client.zcard(order_key) == 1


async def test_create_order_limit_blocks_thirty_first_request(
    redis_client: RedisClient,
) -> None:
    """The 31st create_order call within one hour hits the create_order bucket."""
    ip = "203.0.113.13"
    for _ in range(CREATE_ORDER_MAX_REQUESTS):
        await check_rate_limit(redis_client, ip, CREATE_ORDER_TOOL)

    with pytest.raises(RateLimitExceeded) as exc_info:
        await check_rate_limit(redis_client, ip, CREATE_ORDER_TOOL)

    exc = exc_info.value
    assert exc.limit_type == "create_order"
    assert exc.tool_name == CREATE_ORDER_TOOL
    assert exc.retry_after_seconds >= 1


async def test_read_tools_share_global_bucket(redis_client: RedisClient) -> None:
    """All MCP read tools count toward the same per-IP global window."""
    ip = "203.0.113.14"
    tools = ("kapruka_search_products", "kapruka_get_product", "kapruka_track_order")
    for index in range(GLOBAL_MAX_REQUESTS):
        await check_rate_limit(redis_client, ip, tools[index % len(tools)])

    with pytest.raises(RateLimitExceeded) as exc_info:
        await check_rate_limit(redis_client, ip, "kapruka_list_categories")

    assert exc_info.value.limit_type == "global"
