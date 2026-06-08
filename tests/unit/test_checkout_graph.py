"""Unit tests for checkout sub-graph state machine."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from graphs.checkout_graph import CheckoutGraphDeps, build_checkout_graph
from graphs.checkout_state import (
    CHECKOUT_STEP_ORDER,
    CheckoutState,
    initial_checkout_state,
    next_checkout_step,
    resolve_navigation,
)
from graphs.nodes.checkout_steps import process_checkout_step

COLOMBO = ZoneInfo("Asia/Colombo")
_SESSION_ID = "sess-checkout-graph-001"

_SAMPLE_CART_ITEM = {
    "product_id": "cake00ka002034",
    "quantity": 1,
    "icing_text": None,
    "name": "Chocolate Birthday Cake",
    "price_amount": 4500.0,
    "price_currency": "LKR",
}


def _advance_state(
    *,
    current_step: str,
    step_valid: dict[str, bool] | None = None,
    **fields: Any,
) -> CheckoutState:
    state = initial_checkout_state(
        session_id=_SESSION_ID,
        currency="LKR",
        cart_items=[_SAMPLE_CART_ITEM],
    )
    state["current_step"] = current_step  # type: ignore[typeddict-item]
    state["action"] = "advance"
    state["target_step"] = next_checkout_step(current_step)  # type: ignore[arg-type]
    if step_valid:
        state["step_valid"] = step_valid
    state.update(fields)  # type: ignore[typeddict-item]
    return state


@pytest.mark.parametrize(
    ("current", "target", "action", "expected_step", "allowed"),
    [
        ("cart", "review", "advance", "cart", False),
        ("cart", "delivery_city", "advance", "delivery_city", True),
        ("delivery_date", "recipient", "advance", "recipient", True),
        ("sender", "cart", "back", "cart", True),
        ("recipient", "review", "advance", "recipient", False),
    ],
)
def test_resolve_navigation_blocks_skips_and_allows_sequential(
    current: str,
    target: str,
    action: str,
    expected_step: str,
    allowed: bool,
) -> None:
    """Forward skips are rejected; sequential advance and back are allowed."""
    resolved, ok = resolve_navigation(
        current=current,  # type: ignore[arg-type]
        target=target,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
    )
    assert ok is allowed
    assert resolved == expected_step


@pytest.mark.asyncio
async def test_cannot_jump_from_cart_to_review() -> None:
    """Advancing from cart directly to review is blocked by the navigation gate."""
    graph = build_checkout_graph()
    state: CheckoutState = {
        **initial_checkout_state(session_id=_SESSION_ID, cart_items=[_SAMPLE_CART_ITEM]),
        "action": "advance",
        "target_step": "review",
        "delivery_city": "Colombo 03",
        "delivery_date": "2026-06-10",
        "delivery_address": "123 Galle Road",
        "recipient_name": "Ada",
        "recipient_phone": "0771234567",
        "sender_name": "Bob",
        "sender_anonymous": False,
    }

    result = await graph.ainvoke(state)

    assert result["current_step"] == "cart"
    assert result.get("validation_errors", {}).get("navigation")
    assert not result.get("step_valid", {}).get("review")


@pytest.mark.asyncio
async def test_sequential_advance_through_all_steps() -> None:
    """Valid data advances one step at a time through the full checkout flow."""
    graph = build_checkout_graph()
    fixed_now = datetime(2026, 6, 8, 10, 0, tzinfo=COLOMBO)

    state = initial_checkout_state(
        session_id=_SESSION_ID,
        currency="LKR",
        cart_items=[_SAMPLE_CART_ITEM],
    )

    with patch("lib.utils.timezone.colombo_now", return_value=fixed_now):
        state = await graph.ainvoke(
            {
                **state,
                "action": "advance",
                "target_step": "delivery_city",
            },
        )
        assert state["current_step"] == "delivery_city"
        assert state["step_valid"].get("cart")

        state = await graph.ainvoke(
            {
                **state,
                "delivery_city": "Colombo 03",
                "action": "advance",
                "target_step": "delivery_date",
            },
        )
        assert state["current_step"] == "delivery_date"
        assert state["step_valid"].get("delivery_city")

        state = await graph.ainvoke(
            {
                **state,
                "delivery_date": "2026-06-10",
                "delivery_address": "123 Galle Road",
                "action": "advance",
                "target_step": "recipient",
            },
        )
        assert state["current_step"] == "recipient"
        assert state["step_valid"].get("delivery_date")

        state = await graph.ainvoke(
            {
                **state,
                "recipient_name": "Ada Lovelace",
                "recipient_phone": "+94771234567",
                "action": "advance",
                "target_step": "sender",
            },
        )
        assert state["current_step"] == "sender"
        assert state["step_valid"].get("recipient")

        state = await graph.ainvoke(
            {
                **state,
                "sender_name": "Charles Babbage",
                "sender_anonymous": False,
                "action": "advance",
                "target_step": "review",
            },
        )
        assert state["current_step"] == "review"
        assert state["step_valid"].get("sender")

        state = await graph.ainvoke(
            {
                **state,
                "action": "advance",
                "target_step": "review",
            },
        )
        assert state["current_step"] == "finalize"
        assert state["step_valid"].get("review")
        assert state.get("validation_errors") is None


@pytest.mark.asyncio
async def test_empty_cart_blocks_advance_from_cart() -> None:
    """Cart step validation prevents advancing with an empty cart."""
    graph = build_checkout_graph()
    state: CheckoutState = {
        **initial_checkout_state(session_id=_SESSION_ID, cart_items=[]),
        "action": "advance",
        "target_step": "delivery_city",
    }

    result = await graph.ainvoke(state)

    assert result["current_step"] == "cart"
    assert "cart" in (result.get("validation_errors") or {})
    assert not result.get("step_valid", {}).get("cart")


@pytest.mark.asyncio
async def test_back_action_returns_to_previous_step() -> None:
    """Explicit back navigation moves to the prior step without clearing validity."""
    graph = build_checkout_graph()
    state = _advance_state(
        current_step="recipient",
        step_valid={
            "cart": True,
            "delivery_city": True,
            "delivery_date": True,
        },
        recipient_name="Ada",
        recipient_phone="0771234567",
        delivery_city="Colombo 03",
        delivery_date="2026-06-10",
        delivery_address="123 Galle Road",
    )
    state["action"] = "back"
    state["target_step"] = "delivery_date"

    result = await graph.ainvoke(state)

    assert result["current_step"] == "delivery_date"
    assert result.get("validation_errors") is None


@pytest.mark.asyncio
async def test_process_checkout_step_rejects_wrong_node() -> None:
    """Step node refuses to run when current_step does not match the node."""
    state = initial_checkout_state(session_id=_SESSION_ID, cart_items=[_SAMPLE_CART_ITEM])
    state["current_step"] = "delivery_city"

    delta = await process_checkout_step("cart", state)

    assert "current_step" in (delta.get("validation_errors") or {})


@pytest.mark.asyncio
async def test_cart_loaded_from_redis_when_deps_provided() -> None:
    """Cart step hydrates items from Redis when a client is injected."""
    import fakeredis.aioredis

    from lib.redis.cart import add_item
    from lib.redis.client import RedisClient

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    redis_client = RedisClient("redis://localhost:6379/0", client=fake)
    await add_item(
        redis_client,
        _SESSION_ID,
        product_id="cake00ka002034",
        name="Chocolate Birthday Cake",
        price_amount=4500.0,
        price_currency="LKR",
    )

    graph = build_checkout_graph(deps=CheckoutGraphDeps(redis_client=redis_client))
    state: CheckoutState = {
        **initial_checkout_state(session_id=_SESSION_ID),
        "action": "advance",
        "target_step": "delivery_city",
    }

    result = await graph.ainvoke(state)

    assert result["current_step"] == "delivery_city"
    assert len(result.get("cart_items") or []) == 1


def test_checkout_step_order_matches_prd() -> None:
    """Checkout steps follow the PRD-defined sequence."""
    assert CHECKOUT_STEP_ORDER == (
        "cart",
        "delivery_city",
        "delivery_date",
        "recipient",
        "sender",
        "review",
        "finalize",
    )
