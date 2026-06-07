"""Integration tests for checkout finalize (kapruka_create_order) step."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest

from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.checkout_graph import CheckoutGraphDeps, build_checkout_graph
from graphs.checkout_state import initial_checkout_state
from graphs.nodes.run_checkout_graph import run_checkout_graph
from graphs.state import AgentState
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.create_order import TOOL_NAME as CREATE_ORDER_TOOL
from lib.redis.cart import add_item
from lib.redis.client import RedisClient
from lib.redis.order import get_pending_order

_CLIENT_IP = "203.0.113.42"
_SESSION_ID = "sess-checkout-finalize-001"

_CREATE_ORDER_JSON: dict[str, Any] = {
    "checkout_url": "https://www.kapruka.com/checkout/pay/abc123",
    "order_ref": "ORD-20260608-7823",
    "summary": {
        "items_total": 9000.0,
        "delivery_fee": 350.0,
        "addons_total": 0.0,
        "grand_total": 9350.0,
        "currency": "LKR",
    },
    "expires_at": "2026-06-08T12:30:00+05:30",
}

_SAMPLE_CART_ITEM = {
    "product_id": "cake00ka002034",
    "quantity": 2,
    "icing_text": "Happy Birthday",
    "name": "Chocolate Birthday Cake",
    "price_amount": 4500.0,
    "price_currency": "LKR",
}


def _full_finalize_state() -> dict[str, Any]:
    return {
        **initial_checkout_state(
            session_id=_SESSION_ID,
            currency="LKR",
            cart_items=[_SAMPLE_CART_ITEM],
        ),
        "current_step": "finalize",
        "delivery_city": "Colombo 03",
        "delivery_date": "2026-06-10",
        "delivery_address": "123 Galle Road",
        "delivery_location_type": "house",
        "delivery_instructions": "Ring the bell twice",
        "recipient_name": "Ada Lovelace",
        "recipient_phone": "+94771234567",
        "sender_name": "Charles Babbage",
        "sender_anonymous": False,
        "gift_message": "With love",
        "step_valid": {
            "cart": True,
            "delivery_city": True,
            "delivery_date": True,
            "recipient": True,
            "sender": True,
            "review": True,
        },
        "action": "advance",
        "target_step": "finalize",
    }


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
def mcp_client() -> MCPHttpClient:
    client = AsyncMock(spec=MCPHttpClient)
    client.call_tool = AsyncMock(return_value=json.dumps(_CREATE_ORDER_JSON))
    return client


@pytest.fixture
def kapruka_service(redis_client: RedisClient, mcp_client: MCPHttpClient) -> KaprukaService:
    return KaprukaService(redis_client, mcp_client)


@pytest.mark.asyncio
async def test_finalize_step_returns_checkout_url_with_mocked_mcp(
    redis_client: RedisClient,
    kapruka_service: KaprukaService,
    mcp_client: MCPHttpClient,
) -> None:
    """Finalize node calls create_order and exposes checkout_url in graph state."""
    await add_item(
        redis_client,
        _SESSION_ID,
        product_id=_SAMPLE_CART_ITEM["product_id"],
        name=_SAMPLE_CART_ITEM["name"],
        price_amount=_SAMPLE_CART_ITEM["price_amount"],
        price_currency=_SAMPLE_CART_ITEM["price_currency"],
        quantity=_SAMPLE_CART_ITEM["quantity"],
        icing_text=_SAMPLE_CART_ITEM["icing_text"],
    )
    graph = build_checkout_graph(
        deps=CheckoutGraphDeps(
            redis_client=redis_client,
            kapruka_service=kapruka_service,
            client_ip=_CLIENT_IP,
        ),
    )

    result = await graph.ainvoke(_full_finalize_state())

    assert result["current_step"] == "finalize"
    assert result.get("step_valid", {}).get("finalize") is True
    assert result.get("checkout_url") == _CREATE_ORDER_JSON["checkout_url"]
    assert result.get("order_ref") == _CREATE_ORDER_JSON["order_ref"]
    assert result.get("expires_at") == _CREATE_ORDER_JSON["expires_at"]

    payment_html = result.get("response_html") or ""
    assert 'data-testid="checkout-payment-cta"' in payment_html
    assert _CREATE_ORDER_JSON["order_ref"] in payment_html
    assert "Rs. 9,350" in payment_html

    mcp_client.call_tool.assert_awaited_once()
    assert mcp_client.call_tool.await_args.args[0] == CREATE_ORDER_TOOL

    pending = await get_pending_order(redis_client, _SESSION_ID)
    assert pending is not None
    assert pending.order_ref == _CREATE_ORDER_JSON["order_ref"]
    assert pending.expires_at == _CREATE_ORDER_JSON["expires_at"]


@pytest.mark.asyncio
async def test_run_checkout_graph_passes_checkout_url_in_tool_results(
    redis_client: RedisClient,
    kapruka_service: KaprukaService,
) -> None:
    """Main graph checkout payload includes checkout_url after finalize."""
    await add_item(
        redis_client,
        _SESSION_ID,
        product_id=_SAMPLE_CART_ITEM["product_id"],
        name=_SAMPLE_CART_ITEM["name"],
        price_amount=_SAMPLE_CART_ITEM["price_amount"],
        price_currency=_SAMPLE_CART_ITEM["price_currency"],
        quantity=_SAMPLE_CART_ITEM["quantity"],
        icing_text=_SAMPLE_CART_ITEM["icing_text"],
    )

    from app.templating import render_payment_cta
    from lib.checkout.payment import payment_cta_from_finalize

    payment_context = payment_cta_from_finalize(
        checkout_url=_CREATE_ORDER_JSON["checkout_url"],
        order_ref=_CREATE_ORDER_JSON["order_ref"],
        order_summary=_CREATE_ORDER_JSON["summary"],
        expires_at=_CREATE_ORDER_JSON["expires_at"],
        currency="LKR",
    )
    assert payment_context is not None
    payment_html = render_payment_cta(payment=payment_context)

    finalize_result = {
        **_full_finalize_state(),
        "step_valid": {**_full_finalize_state()["step_valid"], "finalize": True},
        "checkout_url": _CREATE_ORDER_JSON["checkout_url"],
        "order_ref": _CREATE_ORDER_JSON["order_ref"],
        "expires_at": _CREATE_ORDER_JSON["expires_at"],
        "order_summary": _CREATE_ORDER_JSON["summary"],
        "response_html": payment_html,
    }

    state: AgentState = {
        "messages": [],
        "session_id": _SESSION_ID,
        "intent": "checkout",
        "currency": "LKR",
    }

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=finalize_result)

    with patch(
        "graphs.nodes.run_checkout_graph.get_checkout_graph",
        return_value=mock_graph,
    ):
        result = await run_checkout_graph(
            state,
            redis_client=redis_client,
            kapruka_service=kapruka_service,
            client_ip=_CLIENT_IP,
        )

    payload = result["tool_results"][CHECKOUT_TOOL_KEY]
    assert payload["checkout_url"] == _CREATE_ORDER_JSON["checkout_url"]
    assert payload["order_ref"] == _CREATE_ORDER_JSON["order_ref"]
    assert 'data-testid="checkout-payment-cta"' in payload["payment_cta_html"]
    assert payload["review_html"] is None
    assert result["checkout_state"] == "finalize"
    assert result["model_tier"] == "pro"
