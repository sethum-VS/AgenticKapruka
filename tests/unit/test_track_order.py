"""Unit tests for kapruka_track_order wrapper."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from lib.kapruka.errors import KaprukaNotFoundError
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.tools.track_order import TOOL_NAME, track_order
from lib.kapruka.types import TrackOrderOutput, TrackOrderRecipient

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
    "special_instructions": "Ring the bell twice",
    "progress": [
        {"step": "received", "timestamp": "June 5, 2026 10:00 AM"},
        {"step": "confirmed", "timestamp": "June 5, 2026 11:30 AM"},
        {"step": "shipped", "timestamp": "June 6, 2026 08:00 AM"},
    ],
    "live_tracking_available": True,
    "has_delivery_video": False,
    "has_delivery_photo": True,
    "items": [
        {
            "product_id": "cake00ka002034",
            "name": "Chocolate Fudge Cake",
            "quantity": 1,
            "selling_price": 4500.0,
        }
    ],
}


@pytest.fixture
def mcp_client() -> MCPHttpClient:
    client = AsyncMock(spec=MCPHttpClient)
    client.call_tool = AsyncMock(return_value=json.dumps(_TRACK_ORDER_JSON))
    return client


async def test_track_order_parses_response_fields(mcp_client: MCPHttpClient) -> None:
    """Mocked MCP JSON maps to typed TrackOrderOutput with status, events, and ETA."""
    result = await track_order(mcp_client, order_number="VIMP34456CB2")

    assert result.order_number == "VIMP34456CB2"
    assert result.status == "shipped"
    assert result.status_display == "Out for Delivery"
    assert result.delivery_date == "June 7, 2026"
    assert len(result.progress) == 3
    assert result.progress[0].step == "received"
    assert result.progress[2].step == "shipped"
    assert result.recipient.name == "Ada Lovelace"
    assert result.recipient.city == "Colombo 03"
    assert result.live_tracking_available is True
    assert result.has_delivery_photo is True
    assert len(result.items) == 1
    assert result.items[0].name == "Chocolate Fudge Cake"


async def test_track_order_forces_response_format_json(
    mcp_client: MCPHttpClient,
) -> None:
    """MCP params always include response_format=json."""
    await track_order(mcp_client, order_number="VIMP34456CB2")

    mcp_client.call_tool.assert_awaited_once_with(  # type: ignore[attr-defined]
        TOOL_NAME,
        {
            "order_number": "VIMP34456CB2",
            "response_format": "json",
        },
    )


async def test_track_order_validates_order_number_before_mcp_call(
    mcp_client: MCPHttpClient,
) -> None:
    """TrackOrderInput rejects order_number shorter than 4 characters."""
    with pytest.raises(ValidationError):
        await track_order(mcp_client, order_number="ABC")

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]


async def test_track_order_raises_order_not_found_mcp_error(
    mcp_client: MCPHttpClient,
) -> None:
    """order_not_found MCP errors raise KaprukaNotFoundError."""
    mcp_client.call_tool = AsyncMock(  # type: ignore[method-assign]
        return_value="Error (order_not_found): No order found with that number"
    )

    with pytest.raises(KaprukaNotFoundError) as exc_info:
        await track_order(mcp_client, order_number="VIMP99999ZZ9")

    assert exc_info.value.code == "order_not_found"


def test_track_order_output_coerces_mcp_value_currency_shape() -> None:
    """MCP Money payloads with value/currency coerce to a formatted display string."""
    payload = {**_TRACK_ORDER_JSON, "amount": {"value": "4970", "currency": "LKR"}}
    result = TrackOrderOutput.model_validate(payload)
    assert result.amount == "LKR 4,970"


def test_track_order_output_coerces_amount_currency_shape() -> None:
    """MCP Money payloads with amount/currency coerce to a formatted display string."""
    payload = {**_TRACK_ORDER_JSON, "amount": {"amount": 15500.0, "currency": "LKR"}}
    result = TrackOrderOutput.model_validate(payload)
    assert result.amount == "LKR 15,500"


def test_track_order_output_amount_string_passthrough() -> None:
    """Legacy string amounts from MCP pass through unchanged."""
    result = TrackOrderOutput.model_validate(_TRACK_ORDER_JSON)
    assert result.amount == "15500.00"


async def test_track_order_parses_money_shaped_amount_from_mcp(
    mcp_client: MCPHttpClient,
) -> None:
    """Live MCP JSON with value/currency amount maps without validation crash."""
    money_json = {**_TRACK_ORDER_JSON, "amount": {"value": "4970", "currency": "LKR"}}
    mcp_client.call_tool = AsyncMock(return_value=json.dumps(money_json))  # type: ignore[method-assign]

    result = await track_order(mcp_client, order_number="VIMP34456CB2")

    assert result.amount == "LKR 4,970"


def test_track_order_recipient_strips_html_from_phone() -> None:
    """Eval B-06: Kapruka API phone field may leak HTML tags."""
    recipient = TrackOrderRecipient.model_validate(
        {
            "name": "Test User",
            "phone": "0716608447<BR",
            "address": "123 Main St",
            "city": "Colombo",
        }
    )
    assert recipient.phone == "0716608447"


def test_track_order_output_strips_html_from_recipient_phone() -> None:
    """Full track-order payload coerces dirty recipient phone before render."""
    payload = {
        **_TRACK_ORDER_JSON,
        "recipient": {**_TRACK_ORDER_JSON["recipient"], "phone": "0716608447<BR>"},
    }
    result = TrackOrderOutput.model_validate(payload)
    assert result.recipient.phone == "0716608447"
