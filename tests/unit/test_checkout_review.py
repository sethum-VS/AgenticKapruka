"""Unit tests for checkout review step and order summary partial."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from app.templating import render_checkout_review
from graphs.checkout_graph import build_checkout_graph
from graphs.checkout_state import initial_checkout_state
from graphs.model_router import PRO_MODEL, select_model, select_model_tier
from graphs.nodes.run_checkout_graph import run_checkout_graph
from graphs.state import AgentState
from lib.checkout.review import CheckoutReviewContext, review_context_from_checkout_state

COLOMBO = ZoneInfo("Asia/Colombo")
_SESSION_ID = "sess-checkout-review-001"

_SAMPLE_CART_ITEM = {
    "product_id": "cake00ka002034",
    "quantity": 2,
    "icing_text": "Happy Birthday",
    "name": "Chocolate Birthday Cake",
    "price_amount": 4500.0,
    "price_currency": "LKR",
}


def _full_review_state(**overrides: Any) -> dict[str, Any]:
    base = {
        **initial_checkout_state(
            session_id=_SESSION_ID,
            currency="LKR",
            cart_items=[_SAMPLE_CART_ITEM],
        ),
        "current_step": "review",
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
        },
    }
    base.update(overrides)
    return base


def test_review_context_from_checkout_state_builds_summary() -> None:
    context = review_context_from_checkout_state(_full_review_state())  # type: ignore[arg-type]
    assert context is not None
    assert context.item_count == 2
    assert context.items_subtotal == 9000.0
    assert context.sender_display == "Charles Babbage"


def test_review_context_sender_anonymous_display() -> None:
    state = _full_review_state(sender_anonymous=True)
    context = review_context_from_checkout_state(state)  # type: ignore[arg-type]
    assert context is not None
    assert context.sender_display == "Anonymous"


def test_render_checkout_review_includes_all_sections() -> None:
    context = CheckoutReviewContext(
        cart_items=[_SAMPLE_CART_ITEM],
        currency="LKR",
        delivery_address="123 Galle Road",
        delivery_city="Colombo 03",
        delivery_location_type="house",
        delivery_date="2026-06-10",
        delivery_instructions="Ring the bell twice",
        recipient_name="Ada Lovelace",
        recipient_phone="+94771234567",
        sender_name="Charles Babbage",
        sender_anonymous=False,
        gift_message="With love",
    )
    html = render_checkout_review(review=context)

    assert 'data-testid="checkout-review"' in html
    assert "Chocolate Birthday Cake" in html
    assert "Rs. 9,000" in html
    assert "123 Galle Road" in html
    assert "Colombo 03" in html
    assert "Ada Lovelace" in html
    assert "+94771234567" in html
    assert "Charles Babbage" in html
    assert "With love" in html
    assert 'data-testid="checkout-review-delivery"' in html
    assert 'data-testid="checkout-review-recipient"' in html
    assert 'data-testid="checkout-review-sender"' in html


@pytest.mark.asyncio
async def test_review_step_renders_response_html() -> None:
    graph = build_checkout_graph()
    fixed_now = datetime(2026, 6, 8, 10, 0, tzinfo=COLOMBO)

    with patch("lib.utils.timezone.colombo_now", return_value=fixed_now):
        result = await graph.ainvoke(
            {
                **_full_review_state(),
                "action": "advance",
                "target_step": "review",
            },
        )

    assert result["current_step"] == "review"
    assert result["step_valid"].get("review")
    review_html = result.get("response_html") or ""
    assert 'data-testid="checkout-review"' in review_html
    assert "Chocolate Birthday Cake" in review_html


def test_model_router_returns_pro_when_checkout_state_is_review() -> None:
    state: AgentState = {
        "messages": [],
        "session_id": _SESSION_ID,
        "checkout_state": "review",
    }
    assert select_model_tier(state) == "pro"
    assert select_model(state) == PRO_MODEL


@pytest.mark.asyncio
async def test_run_checkout_graph_sets_model_tier_pro_on_review() -> None:
    review_result = {
        "current_step": "review",
        "cart_items": [_SAMPLE_CART_ITEM],
        "step_valid": {"review": True},
        "response_html": '<section data-testid="checkout-review">Review</section>',
    }
    state: AgentState = {
        "messages": [],
        "session_id": _SESSION_ID,
        "intent": "checkout",
        "currency": "LKR",
    }

    with patch("graphs.nodes.run_checkout_graph.get_checkout_graph") as mock_get_graph:
        mock_graph = mock_get_graph.return_value
        mock_graph.ainvoke = AsyncMock(return_value=review_result)
        result = await run_checkout_graph(state, redis_client=None)

    assert result["checkout_state"] == "review"
    assert result["model_tier"] == "pro"
    payload = result["tool_results"]["checkout"]
    assert 'data-testid="checkout-review"' in payload["review_html"]
