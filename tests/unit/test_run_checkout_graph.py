"""Unit tests for run_checkout_graph node."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.nodes.run_checkout_graph import run_checkout_graph
from graphs.state import AgentState
from lib.redis.cart import add_item
from lib.redis.client import RedisClient

_SESSION_ID = "sess-run-checkout-001"


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.mark.asyncio
async def test_run_checkout_graph_hydrates_redis_cart(redis_client: RedisClient) -> None:
    await add_item(
        redis_client,
        _SESSION_ID,
        product_id="cake00ka002034",
        name="Chocolate Birthday Cake",
        price_amount=4500.0,
        quantity=1,
    )

    state: AgentState = {
        "messages": [],
        "session_id": _SESSION_ID,
        "intent": "checkout",
        "currency": "LKR",
    }

    result = await run_checkout_graph(state, redis_client=redis_client)

    assert result["checkout_state"] == "delivery_city"
    payload = result["tool_results"][CHECKOUT_TOOL_KEY]
    assert payload["step_valid"]["cart"] is True
    assert len(payload["cart_items"]) == 1
    assert payload["cart_items"][0]["name"] == "Chocolate Birthday Cake"


@pytest.mark.asyncio
async def test_run_checkout_graph_empty_cart_reports_validation(redis_client: RedisClient) -> None:
    state: AgentState = {
        "messages": [],
        "session_id": _SESSION_ID,
        "intent": "checkout",
    }

    result = await run_checkout_graph(state, redis_client=redis_client)

    assert result["checkout_state"] == "cart"
    payload = result["tool_results"][CHECKOUT_TOOL_KEY]
    assert payload["cart_items"] == []
    assert "cart" in (payload.get("validation_errors") or {})


@pytest.mark.asyncio
async def test_run_checkout_graph_advances_with_persisted_state(redis_client: RedisClient) -> None:
    """Second checkout turn parses delivery city from chat and advances."""
    from lib.redis.checkout import save_checkout_session

    await add_item(
        redis_client,
        _SESSION_ID,
        product_id="cake00ka002034",
        name="Chocolate Birthday Cake",
        price_amount=4500.0,
        quantity=1,
    )
    cart_state = {
        "session_id": _SESSION_ID,
        "current_step": "delivery_city",
        "step_valid": {"cart": True},
        "currency": "LKR",
    }
    await save_checkout_session(redis_client, _SESSION_ID, cart_state)  # type: ignore[arg-type]

    from langchain_core.messages import HumanMessage

    state: AgentState = {
        "messages": [HumanMessage(content="Colombo 03")],
        "session_id": _SESSION_ID,
        "intent": "checkout",
        "currency": "LKR",
    }

    result = await run_checkout_graph(state, redis_client=redis_client)

    assert result["checkout_state"] == "delivery_date"
    payload = result["tool_results"][CHECKOUT_TOOL_KEY]
    assert payload["delivery_city"] == "Colombo 03"
    assert payload["step_valid"].get("delivery_city") is True


@pytest.mark.asyncio
async def test_run_checkout_graph_ignores_duplicate_proceed_at_delivery_city(
    redis_client: RedisClient,
) -> None:
    """Second cart-drawer proceed must not treat the phrase as a city name."""
    from langchain_core.messages import HumanMessage
    from lib.chat.intent_heuristics import PROCEED_CHECKOUT_MESSAGE
    from lib.redis.checkout import save_checkout_session

    await add_item(
        redis_client,
        _SESSION_ID,
        product_id="cake00ka002034",
        name="Chocolate Birthday Cake",
        price_amount=4500.0,
        quantity=1,
    )
    await save_checkout_session(
        redis_client,
        _SESSION_ID,
        {
            "session_id": _SESSION_ID,
            "current_step": "delivery_city",
            "step_valid": {"cart": True},
            "currency": "LKR",
        },  # type: ignore[arg-type]
    )

    state: AgentState = {
        "messages": [HumanMessage(content=PROCEED_CHECKOUT_MESSAGE)],
        "session_id": _SESSION_ID,
        "intent": "checkout",
        "currency": "LKR",
    }

    result = await run_checkout_graph(state, redis_client=redis_client)

    assert result["checkout_state"] == "delivery_city"
    payload = result["tool_results"][CHECKOUT_TOOL_KEY]
    assert payload.get("delivery_city") is None
    assert not payload.get("validation_errors")
