"""Unit tests for kapruka_create_order wrapper."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from lib.kapruka.errors import KaprukaValidationError
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.tools.create_order import TOOL_NAME, create_order
from lib.kapruka.types import CartItem, Delivery, Recipient, Sender
from lib.redis.cache import is_cacheable_tool

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

_CART = [
    CartItem(
        product_id="cake00ka002034",
        quantity=1,
        icing_text="Happy Birthday Mom",
    )
]
_RECIPIENT = Recipient(name="Ada Lovelace", phone="0771234567")
_DELIVERY = Delivery(
    address="123 Galle Road",
    city="Colombo 03",
    date="2026-06-10",
)
_SENDER = Sender(name="Charles Babbage", anonymous=False)


@pytest.fixture
def mcp_client() -> MCPHttpClient:
    client = AsyncMock(spec=MCPHttpClient)
    client.call_tool = AsyncMock(return_value=json.dumps(_CREATE_ORDER_JSON))
    return client


async def test_create_order_parses_response_fields(mcp_client: MCPHttpClient) -> None:
    """Mocked MCP JSON maps to typed CreateOrderResponse with summary totals."""
    result = await create_order(
        mcp_client,
        cart=_CART,
        recipient=_RECIPIENT,
        delivery=_DELIVERY,
        sender=_SENDER,
        gift_message="With love",
        currency="LKR",
    )

    assert result.order_ref == "ORD-20260607-7823"
    assert result.checkout_url == "https://www.kapruka.com/checkout/pay/abc123"
    assert result.expires_at == "2026-06-07T12:30:00+05:30"
    assert result.summary.items_total == 4500.0
    assert result.summary.delivery_fee == 350.0
    assert result.summary.addons_total == 0.0
    assert result.summary.grand_total == 4850.0
    assert result.summary.currency == "LKR"


async def test_create_order_forces_response_format_json(
    mcp_client: MCPHttpClient,
) -> None:
    """MCP params always include response_format=json."""
    await create_order(
        mcp_client,
        cart=_CART,
        recipient=_RECIPIENT,
        delivery=_DELIVERY,
        sender=_SENDER,
    )

    mcp_client.call_tool.assert_awaited_once()  # type: ignore[attr-defined]
    call_args = mcp_client.call_tool.await_args  # type: ignore[attr-defined]
    assert call_args.args[0] == TOOL_NAME
    params = call_args.args[1]
    assert params["response_format"] == "json"
    assert params["cart"][0]["product_id"] == "cake00ka002034"
    assert params["cart"][0]["icing_text"] == "Happy Birthday Mom"
    assert params["recipient"]["name"] == "Ada Lovelace"
    assert params["delivery"]["city"] == "Colombo 03"
    assert params["sender"]["name"] == "Charles Babbage"
    assert params["sender"]["anonymous"] is False


def test_create_order_bypasses_read_cache() -> None:
    """kapruka_create_order is never a cacheable read tool."""
    assert is_cacheable_tool(TOOL_NAME) is False


async def test_create_order_validates_cart_size_before_mcp_call(
    mcp_client: MCPHttpClient,
) -> None:
    """CreateOrderInput rejects empty cart (min 1 item)."""
    with pytest.raises(ValidationError):
        await create_order(
            mcp_client,
            cart=[],
            recipient=_RECIPIENT,
            delivery=_DELIVERY,
            sender=_SENDER,
        )

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]


async def test_create_order_validates_quantity_before_mcp_call(
    mcp_client: MCPHttpClient,
) -> None:
    """CartItem rejects quantity outside 1–99."""
    with pytest.raises(ValidationError):
        await create_order(
            mcp_client,
            cart=[CartItem(product_id="cake00ka002034", quantity=100)],
            recipient=_RECIPIENT,
            delivery=_DELIVERY,
            sender=_SENDER,
        )

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]


async def test_create_order_validates_gift_message_before_mcp_call(
    mcp_client: MCPHttpClient,
) -> None:
    """gift_message over 300 characters fails validation."""
    with pytest.raises(ValidationError):
        await create_order(
            mcp_client,
            cart=_CART,
            recipient=_RECIPIENT,
            delivery=_DELIVERY,
            sender=_SENDER,
            gift_message="x" * 301,
        )

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]


async def test_create_order_rejects_icing_text_on_non_cake(
    mcp_client: MCPHttpClient,
) -> None:
    """icing_text is only allowed on cake product IDs."""
    with pytest.raises(ValueError, match="icing_text is only allowed for cake"):
        await create_order(
            mcp_client,
            cart=[
                CartItem(
                    product_id="flower00ka001",
                    quantity=1,
                    icing_text="Not for flowers",
                )
            ],
            recipient=_RECIPIENT,
            delivery=_DELIVERY,
            sender=_SENDER,
        )

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]


async def test_create_order_raises_empty_cart_mcp_error(
    mcp_client: MCPHttpClient,
) -> None:
    """empty_cart MCP errors raise KaprukaValidationError."""
    mcp_client.call_tool = AsyncMock(  # type: ignore[method-assign]
        return_value="Error (empty_cart): Cart must contain at least one item"
    )

    with pytest.raises(KaprukaValidationError) as exc_info:
        await create_order(
            mcp_client,
            cart=_CART,
            recipient=_RECIPIENT,
            delivery=_DELIVERY,
            sender=_SENDER,
        )

    assert exc_info.value.code == "empty_cart"
