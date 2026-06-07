"""Typed wrapper for kapruka_create_order MCP tool."""

from __future__ import annotations

import json

from lib.kapruka.errors import parse_mcp_error
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.types import (
    CartItem,
    CreateOrderInput,
    CreateOrderResponse,
    Delivery,
    Recipient,
    Sender,
)
from lib.redis.cache import is_cacheable_tool

TOOL_NAME = "kapruka_create_order"


def _is_cake_product(product_id: str) -> bool:
    """Return True when product_id matches Kapruka cake SKU prefixes."""
    return product_id.upper().startswith("CAKE")


def _validate_cart_icing_text(cart: list[CartItem]) -> None:
    """icing_text is allowed only on cake products (max 120 chars via CartItem)."""
    for item in cart:
        if item.icing_text is not None and not _is_cake_product(item.product_id):
            msg = (
                f"icing_text is only allowed for cake products; "
                f"got icing_text on {item.product_id!r}"
            )
            raise ValueError(msg)


async def create_order(
    client: MCPHttpClient,
    *,
    cart: list[CartItem],
    recipient: Recipient,
    delivery: Delivery,
    sender: Sender,
    gift_message: str | None = None,
    currency: str = "LKR",
) -> CreateOrderResponse:
    """Create a Kapruka guest-checkout order via MCP; never read-cached."""
    _validate_cart_icing_text(cart)

    order_input = CreateOrderInput(
        cart=cart,
        recipient=recipient,
        delivery=delivery,
        sender=sender,
        gift_message=gift_message,
        currency=currency,
        response_format="json",
    )
    params = order_input.model_dump(mode="json", exclude_none=True)
    params["response_format"] = "json"

    assert not is_cacheable_tool(TOOL_NAME)

    raw = await client.call_tool(TOOL_NAME, params)
    text = raw.strip()

    parse_mcp_error(text)
    payload = json.loads(text)
    return CreateOrderResponse.model_validate(payload)
