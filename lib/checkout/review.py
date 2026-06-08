"""Checkout review summary context for order confirmation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CheckoutReviewContext:
    """Order summary fields rendered in templates/checkout/review.html."""

    cart_items: list[dict[str, Any]]
    currency: str
    delivery_address: str
    delivery_city: str
    delivery_location_type: str
    delivery_date: str
    delivery_instructions: str | None
    recipient_name: str
    recipient_phone: str
    sender_name: str
    sender_anonymous: bool
    gift_message: str | None

    @property
    def items_subtotal(self) -> float:
        total = 0.0
        for item in self.cart_items:
            qty = int(item.get("quantity") or 1)
            price = float(item.get("price_amount") or 0.0)
            total += price * qty
        return total

    @property
    def sender_display(self) -> str:
        if self.sender_anonymous:
            return "Anonymous"
        return self.sender_name

    @property
    def item_count(self) -> int:
        return sum(int(item.get("quantity") or 1) for item in self.cart_items)


def review_context_from_checkout_state(
    state: Mapping[str, Any],
) -> CheckoutReviewContext | None:
    """Build review context when checkout state has enough data for the summary."""
    cart_items = list(state.get("cart_items") or [])
    if not cart_items:
        return None

    address = (state.get("delivery_address") or "").strip()
    city = (state.get("delivery_city") or "").strip()
    date_value = (state.get("delivery_date") or "").strip()
    recipient_name = (state.get("recipient_name") or "").strip()
    recipient_phone = (state.get("recipient_phone") or "").strip()
    sender_name = (state.get("sender_name") or "").strip()

    if not all([address, city, date_value, recipient_name, recipient_phone, sender_name]):
        return None

    default_currency = cart_items[0].get("price_currency") or "LKR"
    currency = str(state.get("currency") or default_currency)
    instructions = state.get("delivery_instructions")
    instructions_text = None
    if isinstance(instructions, str) and instructions.strip():
        instructions_text = instructions.strip()
    gift = state.get("gift_message")
    gift_text = None
    if isinstance(gift, str) and gift.strip():
        gift_text = gift.strip()

    return CheckoutReviewContext(
        cart_items=cart_items,
        currency=currency,
        delivery_address=address,
        delivery_city=city,
        delivery_location_type=(state.get("delivery_location_type") or "house").strip(),
        delivery_date=date_value,
        delivery_instructions=instructions_text,
        recipient_name=recipient_name,
        recipient_phone=recipient_phone,
        sender_name=sender_name,
        sender_anonymous=bool(state.get("sender_anonymous")),
        gift_message=gift_text,
    )
