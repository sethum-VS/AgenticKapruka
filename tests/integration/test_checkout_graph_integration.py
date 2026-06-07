"""Integration tests for checkout sub-graph wired into the main shopping graph."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest
from langchain_core.messages import HumanMessage

from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.nodes.analyze_intent import PROCEED_CHECKOUT_MESSAGE, IntentClassification
from graphs.shopping_graph import ShoppingGraphDeps, build_shopping_graph, initial_shopping_state
from graphs.state import AgentState
from lib.redis.cart import add_item
from lib.redis.client import RedisClient

_CLIENT_IP = "203.0.113.42"
_SESSION_ID = "sess-checkout-integration-001"

_SAMPLE_ITEM = {
    "product_id": "cake00ka002034",
    "name": "Chocolate Birthday Cake",
    "price_amount": 4500.0,
    "price_currency": "LKR",
    "quantity": 2,
}


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


def _mock_genai_checkout_client() -> MagicMock:
    mock_client = MagicMock()
    intent_response = MagicMock()
    intent_response.parsed = IntentClassification(intent="checkout")
    intent_response.text = '{"intent": "checkout"}'
    mock_client.models.generate_content.return_value = intent_response
    return mock_client


@pytest.mark.asyncio
async def test_checkout_flow_starts_from_cart_with_redis_items(
    redis_client: RedisClient,
) -> None:
    """Checkout intent routes to sub-graph and hydrates cart lines from Redis."""
    await add_item(
        redis_client,
        _SESSION_ID,
        product_id=_SAMPLE_ITEM["product_id"],
        name=_SAMPLE_ITEM["name"],
        price_amount=_SAMPLE_ITEM["price_amount"],
        price_currency=_SAMPLE_ITEM["price_currency"],
        quantity=_SAMPLE_ITEM["quantity"],
    )

    deps = ShoppingGraphDeps(
        redis_client=redis_client,
        kapruka_service=None,
        client_ip=_CLIENT_IP,
        genai_client=_mock_genai_checkout_client(),
    )
    graph = build_shopping_graph(deps=deps)
    state: AgentState = initial_shopping_state(
        message="I want to place my order",
        session_id=_SESSION_ID,
        currency="LKR",
    )

    result = await graph.ainvoke(state)

    assert result["intent"] == "checkout"
    assert result["checkout_state"] == "cart"
    checkout_payload = (result.get("tool_results") or {}).get(CHECKOUT_TOOL_KEY)
    assert isinstance(checkout_payload, dict)
    assert checkout_payload.get("step_valid", {}).get("cart") is True
    cart_items = checkout_payload.get("cart_items")
    assert isinstance(cart_items, list)
    assert len(cart_items) == 1
    assert cart_items[0]["product_id"] == _SAMPLE_ITEM["product_id"]
    assert cart_items[0]["quantity"] == _SAMPLE_ITEM["quantity"]
    assert "check out your 2 cart items" in (result.get("assistant_message") or "").lower()


@pytest.mark.asyncio
async def test_proceed_checkout_message_routes_without_gemini_classification(
    redis_client: RedisClient,
) -> None:
    """Cart drawer proceed trigger classifies checkout deterministically."""
    await add_item(
        redis_client,
        _SESSION_ID,
        product_id=_SAMPLE_ITEM["product_id"],
        name=_SAMPLE_ITEM["name"],
        price_amount=_SAMPLE_ITEM["price_amount"],
        quantity=1,
    )

    genai_client = MagicMock()
    deps = ShoppingGraphDeps(
        redis_client=redis_client,
        client_ip=_CLIENT_IP,
        genai_client=genai_client,
    )
    graph = build_shopping_graph(deps=deps)
    result = await graph.ainvoke(
        initial_shopping_state(
            message=PROCEED_CHECKOUT_MESSAGE,
            session_id=_SESSION_ID,
        ),
    )

    assert result["intent"] == "checkout"
    assert result["checkout_state"] == "cart"
    genai_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_checkout_graph_node_order_via_stream_events(
    redis_client: RedisClient,
) -> None:
    """Checkout intent skips hybrid context and MCP tools."""
    await add_item(
        redis_client,
        _SESSION_ID,
        product_id=_SAMPLE_ITEM["product_id"],
        name=_SAMPLE_ITEM["name"],
        price_amount=_SAMPLE_ITEM["price_amount"],
        quantity=1,
    )

    deps = ShoppingGraphDeps(
        redis_client=redis_client,
        client_ip=_CLIENT_IP,
        genai_client=_mock_genai_checkout_client(),
    )
    graph = build_shopping_graph(deps=deps)
    state: AgentState = {
        "messages": [HumanMessage(content="checkout my cart")],
        "session_id": _SESSION_ID,
    }
    config: dict[str, Any] = {"configurable": {"thread_id": "thread-checkout-order"}}

    node_names: list[str] = []
    async for event in graph.astream_events(state, config, version="v2"):
        if event.get("event") == "on_chain_start" and event.get("name") in {
            "load_zep_memory",
            "analyze_intent",
            "retrieve_hybrid_context",
            "call_mcp_tools",
            "run_checkout_graph",
            "generate_response",
            "zep_memory_write",
        }:
            node_names.append(str(event["name"]))

    assert node_names == [
        "load_zep_memory",
        "analyze_intent",
        "run_checkout_graph",
        "generate_response",
        "zep_memory_write",
    ]
