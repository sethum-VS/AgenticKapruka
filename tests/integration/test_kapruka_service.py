"""Integration tests for KaprukaService cache and rate-limit wiring."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.create_order import TOOL_NAME as CREATE_ORDER_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_TOOL
from lib.kapruka.types import CartItem, Delivery, Recipient, Sender
from lib.redis.cache import DEFAULT_CACHE_TTL
from lib.redis.client import RedisClient

_CLIENT_IP = "203.0.113.42"

_SEARCH_JSON: dict[str, Any] = {
    "results": [
        {
            "id": "cake00ka002034",
            "name": "Chocolate Birthday Cake",
            "summary": "Rich chocolate cake.",
            "price": {"amount": 4500.0, "currency": "LKR"},
            "compare_at_price": None,
            "in_stock": True,
            "stock_level": "high",
            "image_url": "https://static2.kapruka.com/cake.jpg",
            "category": {"id": "cat_cakes", "name": "Birthday", "slug": "birthday"},
            "rating": None,
            "ships_internationally": False,
            "url": "https://www.kapruka.com/cake",
        }
    ],
    "next_cursor": None,
    "applied_filters": {"q": "birthday cake", "limit": 10, "in_stock_only": False},
}

_CREATE_ORDER_JSON: dict[str, Any] = {
    "checkout_url": "https://www.kapruka.com/checkout/pay/abc123",
    "order_ref": "ORD-20260607-7823",
    "summary": {
        "items_total": 4500.0,
        "delivery_fee": 350.0,
        "addons_total": 0.0,
        "grand_total": 4850.0,
        "currency": "LKR",
    },
    "expires_at": "2026-06-07T12:30:00+05:30",
}

_CART = [CartItem(product_id="cake00ka002034", quantity=1)]
_RECIPIENT = Recipient(name="Ada Lovelace", phone="0771234567")
_DELIVERY = Delivery(address="123 Galle Road", city="Colombo 03", date="2026-06-10")
_SENDER = Sender(name="Charles Babbage", anonymous=False)


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
def mcp_client() -> MCPHttpClient:
    client = AsyncMock(spec=MCPHttpClient)

    async def _call_tool(tool_name: str, params: dict[str, Any]) -> str:
        if tool_name == SEARCH_TOOL:
            return json.dumps(_SEARCH_JSON)
        if tool_name == CREATE_ORDER_TOOL:
            return json.dumps(_CREATE_ORDER_JSON)
        msg = f"unexpected tool: {tool_name}"
        raise AssertionError(msg)

    client.call_tool = AsyncMock(side_effect=_call_tool)
    return client


@pytest.fixture
def service(redis_client: RedisClient, mcp_client: MCPHttpClient) -> KaprukaService:
    return KaprukaService(redis_client, mcp_client)


async def test_search_products_cache_hit_avoids_second_mcp_call(
    service: KaprukaService,
    mcp_client: MCPHttpClient,
    redis_client: RedisClient,
) -> None:
    """Second identical search returns cached response; MCP is called only once."""
    first = await service.search_products(_CLIENT_IP, q="birthday cake", limit=10)
    second = await service.search_products(_CLIENT_IP, q="birthday cake", limit=10)

    assert first.results[0].id == "cake00ka002034"
    assert second.results[0].id == "cake00ka002034"

    search_calls = [
        call
        for call in mcp_client.call_tool.await_args_list  # type: ignore[attr-defined]
        if call.args[0] == SEARCH_TOOL
    ]
    assert len(search_calls) == 1

    keys = [key async for key in redis_client.client.scan_iter(match=f"{SEARCH_TOOL}:*")]
    assert len(keys) == 1
    ttl = await redis_client.client.ttl(keys[0])
    assert 0 < ttl <= DEFAULT_CACHE_TTL


async def test_create_order_bypasses_cache_on_repeated_calls(
    service: KaprukaService,
    mcp_client: MCPHttpClient,
    redis_client: RedisClient,
) -> None:
    """create_order always hits MCP; no Redis cache entries are written."""
    first = await service.create_order(
        _CLIENT_IP,
        cart=_CART,
        recipient=_RECIPIENT,
        delivery=_DELIVERY,
        sender=_SENDER,
    )
    second = await service.create_order(
        _CLIENT_IP,
        cart=_CART,
        recipient=_RECIPIENT,
        delivery=_DELIVERY,
        sender=_SENDER,
    )

    assert first.order_ref == "ORD-20260607-7823"
    assert second.order_ref == "ORD-20260607-7823"

    create_calls = [
        call
        for call in mcp_client.call_tool.await_args_list  # type: ignore[attr-defined]
        if call.args[0] == CREATE_ORDER_TOOL
    ]
    assert len(create_calls) == 2

    keys = [key async for key in redis_client.client.scan_iter(match=f"{CREATE_ORDER_TOOL}:*")]
    assert keys == []
