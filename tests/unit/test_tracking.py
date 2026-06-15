"""Unit tests for order reference classification and tracking failure copy."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.generate_response import build_agent_tool_error_message, generate_response
from graphs.state import AgentState
from lib.checkout.tracking import (
    build_tracking_failure_message,
    classify_order_reference,
    classify_order_references,
    extract_order_number,
    tracking_output_from_tool_results,
)
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import TrackOrderOutput

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
    "greeting_message": None,
    "special_instructions": None,
    "progress": [],
    "live_tracking_available": False,
    "has_delivery_video": False,
    "has_delivery_photo": False,
    "items": [],
}


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("VIMP34456CB2", "vimp"),
        ("vimp99887ab1", "vimp"),
        ("ORD-20260520-7823", "ord_ref"),
        ("KA123456", "ka_legacy"),
        ("KA-12345678", "ka_legacy"),
        ("", "unknown"),
        ("XYZ", "unknown"),
    ],
)
def test_classify_order_reference(ref: str, expected: str) -> None:
    assert classify_order_reference(ref) == expected


def test_classify_order_references_extracts_multiple_kinds() -> None:
    refs = classify_order_references("track ORD-20260520-7823 or VIMP34456CB2")
    kinds = {kind for _, kind in refs}
    assert "ord_ref" in kinds
    assert "vimp" in kinds


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("where is order VIMP34456CB2?", "VIMP34456CB2"),
        ("Track order KA123456", "KA123456"),
        ("Where is order KA-12345678?", "KA12345678"),
        ("track ORD-20260520-7823 please", None),
    ],
)
def test_extract_order_number(message: str, expected: str | None) -> None:
    assert extract_order_number(message) == expected


def test_tracking_output_from_tool_results_ignores_error_payload() -> None:
    assert (
        tracking_output_from_tool_results(
            {TRACK_ORDER_TOOL: {"error": "order_not_found", "message": "not found"}},
        )
        is None
    )


def test_build_tracking_failure_message_ka_legacy_educates_vimp() -> None:
    message = build_tracking_failure_message(
        order_number="KA123456",
        reference_kind="ka_legacy",
        error_code="order_not_found",
    )
    assert "KA123456" in message
    assert "VIMP34456CB2" in message
    assert "legacy" in message.lower()
    assert "could not find order" in message.lower()


def test_build_agent_tool_error_message_track_order_ka_legacy() -> None:
    message = build_agent_tool_error_message(
        tool=TRACK_ORDER_TOOL,
        raw_message="We could not find an order with that number.",
        error_code="order_not_found",
        order_number="KA123456",
        reference_kind="ka_legacy",
    )
    assert "VIMP34456CB2" in message
    assert "legacy" in message.lower()


@pytest.mark.asyncio
async def test_generate_response_study_turn_8_ka_legacy_educate() -> None:
    """Customer study turn 8: KA legacy → graceful educate, no SSE crash."""
    state: AgentState = {
        "messages": [HumanMessage(content="Where is order KA123456?")],
        "intent": "tracking",
        "tool_results": {
            TRACK_ORDER_TOOL: {
                "error": "order_not_found",
                "message": "We could not find an order with that number.",
            },
        },
    }

    result = await generate_response(state, genai_client=MagicMock())

    assert "VIMP34456CB2" in result["assistant_message"]
    assert "legacy" in result["assistant_message"].lower()
    assert 'data-testid="order-tracking-status"' not in result["response_html"]


@pytest.mark.asyncio
async def test_generate_response_study_turn_9_vimp_status_card() -> None:
    """Customer study turn 9: VIMP → tracking status card."""
    state: AgentState = {
        "messages": [HumanMessage(content="Where is order VIMP34456CB2?")],
        "intent": "tracking",
        "tool_results": {TRACK_ORDER_TOOL: _TRACK_ORDER_JSON},
    }

    result = await generate_response(state, genai_client=MagicMock())

    assert "VIMP34456CB2" in result["assistant_message"]
    assert 'data-testid="order-tracking-status"' in result["response_html"]


@pytest.mark.asyncio
async def test_generate_response_study_turn_10_track_my_order_vimp() -> None:
    """Customer study turn 10: track my order VIMP… → status card."""
    state: AgentState = {
        "messages": [HumanMessage(content="track my order VIMP34456CB2")],
        "intent": "tracking",
        "tool_results": {TRACK_ORDER_TOOL: _TRACK_ORDER_JSON},
    }

    result = await generate_response(state, genai_client=MagicMock())

    assert (
        TrackOrderOutput.model_validate(_TRACK_ORDER_JSON).status_display
        in result["assistant_message"]
    )
    assert 'data-testid="order-tracking-status"' in result["response_html"]


@pytest.mark.asyncio
async def test_generate_response_tracking_tool_error_no_crash() -> None:
    """kapruka_track_order error payload must not raise during generate_response."""
    mock_client = MagicMock()
    state: AgentState = {
        "messages": [HumanMessage(content="check status VIMP99999ZZ9")],
        "intent": "tracking",
        "tool_results": {
            TRACK_ORDER_TOOL: {
                "error": "order_not_found",
                "message": "We could not find an order with that number.",
            },
        },
    }

    result = await generate_response(state, genai_client=mock_client)

    mock_client.models.generate_content.assert_not_called()
    assert "VIMP99999ZZ9" in result["assistant_message"]
    assert 'data-testid="product-carousel"' not in result["response_html"]
