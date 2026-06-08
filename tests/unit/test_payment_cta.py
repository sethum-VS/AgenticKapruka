"""Unit tests for click-to-pay countdown UI component."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import HumanMessage

from app.templating import render_payment_cta
from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.checkout_graph import CheckoutGraphDeps, build_checkout_graph
from graphs.nodes.generate_response import generate_response, render_assistant_html
from graphs.nodes.run_checkout_graph import run_checkout_graph
from graphs.state import AgentState
from lib.checkout.payment import (
    PaymentCtaContext,
    countdown_remaining_seconds,
    format_countdown_mm_ss,
    is_countdown_warning,
    parse_expires_at_iso,
    payment_cta_from_finalize,
)

COLOMBO = ZoneInfo("Asia/Colombo")
_SESSION_ID = "sess-payment-cta-001"

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

_PAYMENT_CONTEXT = PaymentCtaContext(
    checkout_url=_CREATE_ORDER_JSON["checkout_url"],
    order_ref=_CREATE_ORDER_JSON["order_ref"],
    grand_total=9350.0,
    currency="LKR",
    expires_at=_CREATE_ORDER_JSON["expires_at"],
)


def test_payment_cta_from_finalize_builds_context() -> None:
    context = payment_cta_from_finalize(
        checkout_url=_CREATE_ORDER_JSON["checkout_url"],
        order_ref=_CREATE_ORDER_JSON["order_ref"],
        order_summary=_CREATE_ORDER_JSON["summary"],
        expires_at=_CREATE_ORDER_JSON["expires_at"],
        currency="LKR",
    )
    assert context is not None
    assert context.grand_total == 9350.0
    assert context.currency == "LKR"


def test_payment_cta_from_finalize_returns_none_when_fields_missing() -> None:
    assert (
        payment_cta_from_finalize(
            checkout_url="",
            order_ref="ORD-1",
            order_summary=None,
            expires_at="2026-06-08T12:30:00+05:30",
        )
        is None
    )


def test_parse_expires_at_iso_and_countdown_helpers() -> None:
    expires_at = parse_expires_at_iso("2026-06-08T12:30:00+05:30")
    now = datetime(2026, 6, 8, 12, 20, 0, tzinfo=COLOMBO)
    remaining = countdown_remaining_seconds(expires_at, now=now)
    assert remaining == 600
    assert format_countdown_mm_ss(remaining) == "10:00"
    assert is_countdown_warning(remaining) is False
    assert is_countdown_warning(599) is True
    assert format_countdown_mm_ss(0) == "00:00"


def test_render_payment_cta_includes_cta_and_order_details() -> None:
    html = render_payment_cta(payment=_PAYMENT_CONTEXT)

    assert 'data-testid="checkout-payment-cta"' in html
    assert "ORD-20260608-7823" in html
    assert "Rs. 9,350" in html
    assert 'href="https://www.kapruka.com/checkout/pay/abc123"' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html
    assert 'data-testid="checkout-payment-cta-button"' in html
    assert "paymentCountdown(" in html
    assert "2026-06-08T12:30:00+05:30" in html
    assert 'data-testid="checkout-payment-warning"' in html
    assert 'data-testid="checkout-payment-expired"' in html


def test_render_assistant_html_embeds_payment_cta_slot() -> None:
    payment_html = render_payment_cta(payment=_PAYMENT_CONTEXT)
    html = render_assistant_html(
        "Your order is ready.",
        checkout_payment_html=payment_html,
    )
    assert 'data-slot="checkout-payment"' in html
    assert 'data-testid="checkout-payment-cta"' in html


@pytest.mark.asyncio
async def test_finalize_step_renders_payment_cta_html() -> None:
    graph = build_checkout_graph(
        deps=CheckoutGraphDeps(
            redis_client=None,
            kapruka_service=AsyncMock(),
            client_ip="127.0.0.1",
        ),
    )
    finalize_state: dict[str, Any] = {
        "session_id": _SESSION_ID,
        "currency": "LKR",
        "current_step": "finalize",
        "cart_items": [],
        "step_valid": {"review": True},
        "action": "advance",
    }

    with patch(
        "graphs.nodes.checkout_steps.execute_finalize_step",
        new_callable=AsyncMock,
        return_value=(
            True,
            {},
            {
                "checkout_url": _CREATE_ORDER_JSON["checkout_url"],
                "order_ref": _CREATE_ORDER_JSON["order_ref"],
                "expires_at": _CREATE_ORDER_JSON["expires_at"],
                "order_summary": _CREATE_ORDER_JSON["summary"],
            },
        ),
    ):
        result = await graph.ainvoke(finalize_state)

    payment_html = result.get("response_html") or ""
    assert 'data-testid="checkout-payment-cta"' in payment_html
    assert "ORD-20260608-7823" in payment_html


@pytest.mark.asyncio
async def test_generate_response_checkout_finalize_embeds_payment_cta() -> None:
    payment_html = render_payment_cta(payment=_PAYMENT_CONTEXT)
    state: AgentState = {
        "messages": [HumanMessage(content="place my order")],
        "intent": "checkout",
        "checkout_state": "finalize",
        "tool_results": {
            CHECKOUT_TOOL_KEY: {
                "current_step": "finalize",
                "cart_items": [{"product_id": "cake001", "quantity": 1}],
                "checkout_url": _CREATE_ORDER_JSON["checkout_url"],
                "order_ref": _CREATE_ORDER_JSON["order_ref"],
                "payment_cta_html": payment_html,
            },
        },
        "session_id": _SESSION_ID,
    }

    result = await generate_response(state)

    assert 'data-slot="checkout-payment"' in result["response_html"]
    assert 'data-testid="checkout-payment-cta"' in result["response_html"]
    assert "button below" in result["assistant_message"].lower()


@pytest.mark.asyncio
async def test_run_checkout_graph_passes_payment_cta_html_on_finalize() -> None:
    payment_html = render_payment_cta(payment=_PAYMENT_CONTEXT)
    finalize_result = {
        "current_step": "finalize",
        "cart_items": [{"product_id": "cake001", "quantity": 1}],
        "response_html": payment_html,
        "checkout_url": _CREATE_ORDER_JSON["checkout_url"],
        "order_ref": _CREATE_ORDER_JSON["order_ref"],
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
        result = await run_checkout_graph(state, redis_client=None)

    payload = result["tool_results"][CHECKOUT_TOOL_KEY]
    assert 'data-testid="checkout-payment-cta"' in payload["payment_cta_html"]
    assert payload["review_html"] is None
