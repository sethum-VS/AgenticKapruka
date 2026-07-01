"""Unit tests for order tracking chat flow and tracking_status partial."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from app.templating import normalize_html_snapshot, render_tracking_status
from graphs.nodes.call_mcp_tools import call_mcp_tools, select_tool_calls
from graphs.nodes.generate_response import (
    build_tracking_status_html,
    generate_response,
    render_assistant_html,
)
from graphs.state import AgentState
from lib.checkout.tracking import extract_order_number
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import TrackOrderOutput

_CLIENT_IP = "203.0.113.99"

_TRACK_ORDER_JSON: dict[str, Any] = {
    "order_number": "VIMP34456CB2",
    "pnref": "12345678901",
    "status": "shipped",
    "status_display": "Out for Delivery",
    "order_date": "June 5, 2026",
    "delivery_date": "June 7, 2026",
    "shipped_date": "June 6, 2026",
    "amount": "15500.00",
    "payment_method": "Visa",
    "comments": None,
    "recipient": {
        "name": "Ada Lovelace",
        "phone": "0771234567",
        "address": "123 Galle Road",
        "city": "Colombo 03",
    },
    "greeting_message": "Happy Birthday!",
    "special_instructions": None,
    "progress": [
        {"step": "received", "timestamp": "June 5, 2026 10:00 AM"},
        {"step": "confirmed", "timestamp": "June 5, 2026 11:30 AM"},
        {"step": "shipped", "timestamp": "June 6, 2026 08:00 AM"},
    ],
    "live_tracking_available": True,
    "has_delivery_video": False,
    "has_delivery_photo": True,
    "items": [],
}

_TRACK_OUTPUT = TrackOrderOutput.model_validate(_TRACK_ORDER_JSON)


def test_extract_order_number_finds_post_payment_id() -> None:
    assert extract_order_number("where is order VIMP34456CB2?") == "VIMP34456CB2"


def test_extract_order_number_ignores_pre_payment_order_ref() -> None:
    assert extract_order_number("track ORD-20260520-7823 please") is None


def test_render_tracking_status_includes_timeline_steps() -> None:
    html = render_tracking_status(tracking=_TRACK_OUTPUT)
    normalized = normalize_html_snapshot(html)

    assert 'data-testid="order-tracking-status"' in normalized
    assert 'data-testid="tracking-timeline"' in normalized
    assert 'data-step="received"' in normalized
    assert 'data-step="confirmed"' in normalized
    assert 'data-step="shipped"' in normalized
    assert "Out for Delivery" in normalized
    assert "VIMP34456CB2" in normalized


def test_render_tracking_status_recipient_phone_without_html_artifacts() -> None:
    """Eval B-06: tracking partial must not render Kapruka HTML tag leaks in phone."""
    payload = {
        **_TRACK_ORDER_JSON,
        "recipient": {**_TRACK_ORDER_JSON["recipient"], "phone": "0716608447<BR"},
    }
    tracking = TrackOrderOutput.model_validate(payload)
    html = render_tracking_status(tracking=tracking)
    normalized = normalize_html_snapshot(html)

    assert 'data-testid="tracking-recipient-phone"' in normalized
    assert "0716608447" in normalized
    assert "<BR" not in normalized.upper()
    assert "<br" not in normalized


def test_build_tracking_status_html_from_tool_results() -> None:
    html = build_tracking_status_html({TRACK_ORDER_TOOL: _TRACK_ORDER_JSON})
    assert html is not None
    assert 'data-testid="tracking-timeline-step"' in html


def test_build_tracking_status_html_coerces_money_shaped_amount() -> None:
    """Tracking partial renders when MCP returns value/currency amount object."""
    payload = {**_TRACK_ORDER_JSON, "amount": {"value": "4970", "currency": "LKR"}}
    html = build_tracking_status_html({TRACK_ORDER_TOOL: payload})
    assert html is not None
    assert "Rs. 4,970" in html
    assert 'data-testid="tracking-amount"' in html


def test_select_tool_calls_tracking_with_order_number() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="track order VIMP34456CB2")],
        "intent": "tracking",
    }
    selected = select_tool_calls(state)
    assert len(selected) == 1
    assert selected[0]["name"] == TRACK_ORDER_TOOL
    assert selected[0]["args"]["order_number"] == "VIMP34456CB2"


def test_select_tool_calls_tracking_without_order_number_returns_empty() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="where is my order?")],
        "intent": "tracking",
    }
    assert select_tool_calls(state) == []


@pytest.mark.asyncio
async def test_call_mcp_tools_tracking_invokes_track_order() -> None:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.track_order.return_value = _TRACK_OUTPUT

    state: AgentState = {
        "messages": [HumanMessage(content="where is order VIMP34456CB2")],
        "intent": "tracking",
        "session_id": "sess-track-001",
    }

    result = await call_mcp_tools(
        state,
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
    )

    mock_service.track_order.assert_awaited_once_with(
        _CLIENT_IP,
        order_number="VIMP34456CB2",
    )
    assert result["tool_call_count"] == 1
    assert result["tool_results"][TRACK_ORDER_TOOL]["status_display"] == "Out for Delivery"


@pytest.mark.asyncio
async def test_generate_response_tracking_embeds_status_partial() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="track VIMP34456CB2")],
        "intent": "tracking",
        "tool_results": {TRACK_ORDER_TOOL: _TRACK_ORDER_JSON},
    }

    result = await generate_response(state, genai_client=MagicMock())

    assert "VIMP34456CB2" in result["assistant_message"]
    assert 'data-testid="tracking-timeline"' in result["response_html"]
    assert 'data-slot="tracking-status"' in result["response_html"]


@pytest.mark.asyncio
async def test_generate_response_tracking_without_number_skips_llm() -> None:
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="where is my order?")],
        "intent": "tracking",
        "tool_results": {},
    }

    result = await generate_response(state, genai_client=mock_client)

    mock_client.models.generate_content.assert_not_called()
    assert "order number" in result["assistant_message"].lower()
    assert "ORD-" in result["assistant_message"]


def test_render_assistant_html_accepts_tracking_slot() -> None:
    tracking_html = render_tracking_status(tracking=_TRACK_OUTPUT)
    html = render_assistant_html(
        "Status update below.",
        tracking_status_html=tracking_html,
    )
    assert 'data-slot="tracking-status"' in html
    assert 'data-testid="tracking-timeline"' in html
