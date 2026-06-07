"""Build Kapruka create_order payloads from checkout state and Redis cart."""

from __future__ import annotations

from typing import Any

from graphs.checkout_state import CheckoutState
from lib.kapruka.types import CartItem, Delivery, Recipient, Sender


def stored_cart_to_mcp_cart(cart_items: list[dict[str, Any]]) -> list[CartItem]:
    """Map Redis cart line dicts to Kapruka MCP CartItem models."""
    return [
        CartItem(
            product_id=str(row["product_id"]),
            quantity=int(row.get("quantity", 1)),
            icing_text=row.get("icing_text"),
        )
        for row in cart_items
    ]


def build_create_order_from_checkout(
    state: CheckoutState,
    cart_items: list[dict[str, Any]],
) -> tuple[Recipient, Delivery, Sender, list[CartItem], str | None, str]:
    """Assemble create_order arguments from checkout state and cart rows."""
    if not cart_items:
        msg = "Cart is empty."
        raise ValueError(msg)

    recipient = Recipient(
        name=state.get("recipient_name") or "",
        phone=state.get("recipient_phone") or "",
    )
    instructions = (state.get("delivery_instructions") or "").strip() or None
    delivery = Delivery(
        address=state.get("delivery_address") or "",
        city=state.get("delivery_city") or "",
        location_type=state.get("delivery_location_type") or "house",
        date=state.get("delivery_date") or "",
        instructions=instructions,
    )
    sender = Sender(
        name=state.get("sender_name") or "",
        anonymous=bool(state.get("sender_anonymous")),
    )
    cart = stored_cart_to_mcp_cart(cart_items)
    gift_message = (state.get("gift_message") or "").strip() or None
    currency = state.get("currency") or "LKR"
    return recipient, delivery, sender, cart, gift_message, currency
