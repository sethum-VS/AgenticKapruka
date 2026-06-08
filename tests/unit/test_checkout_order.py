"""Unit tests for building Kapruka create_order payloads from checkout state."""

from __future__ import annotations

import pytest

from graphs.checkout_state import initial_checkout_state
from lib.checkout.order import build_create_order_from_checkout, stored_cart_to_mcp_cart
from lib.kapruka.types import CartItem

_CART_ROW = {
    "product_id": "cake00ka002034",
    "quantity": 2,
    "icing_text": "Happy Birthday",
    "name": "Chocolate Birthday Cake",
    "price_amount": 4500.0,
    "price_currency": "LKR",
}


def test_stored_cart_to_mcp_cart_maps_fields() -> None:
    items = stored_cart_to_mcp_cart([_CART_ROW])
    assert len(items) == 1
    assert items[0] == CartItem(
        product_id="cake00ka002034",
        quantity=2,
        icing_text="Happy Birthday",
    )


def test_build_create_order_from_checkout_assembles_models() -> None:
    state = initial_checkout_state(session_id="sess-order-001", currency="LKR")
    state.update(
        {
            "delivery_city": "Colombo 03",
            "delivery_date": "2026-06-10",
            "delivery_address": "123 Galle Road",
            "delivery_location_type": "apartment",
            "delivery_instructions": "Ring twice",
            "recipient_name": "Ada Lovelace",
            "recipient_phone": "+94771234567",
            "sender_name": "Charles Babbage",
            "sender_anonymous": False,
            "gift_message": "With love",
        },
    )

    recipient, delivery, sender, cart, gift_message, currency = build_create_order_from_checkout(
        state,
        [_CART_ROW],
    )

    assert recipient.name == "Ada Lovelace"
    assert recipient.phone == "+94771234567"
    assert delivery.city == "Colombo 03"
    assert delivery.location_type == "apartment"
    assert delivery.instructions == "Ring twice"
    assert sender.name == "Charles Babbage"
    assert sender.anonymous is False
    assert len(cart) == 1
    assert gift_message == "With love"
    assert currency == "LKR"


def test_build_create_order_from_checkout_rejects_empty_cart() -> None:
    state = initial_checkout_state(session_id="sess-order-002")
    with pytest.raises(ValueError, match="empty"):
        build_create_order_from_checkout(state, [])
