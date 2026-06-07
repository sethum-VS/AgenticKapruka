"""Unit tests for Kapruka MCP Pydantic types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lib.kapruka.types import (
    CartItem,
    CreateOrderInput,
    Delivery,
    Recipient,
    Sender,
)


def _minimal_create_order_input(
    *,
    gift_message: str | None = None,
    icing_text: str | None = None,
) -> CreateOrderInput:
    return CreateOrderInput(
        cart=[
            CartItem(
                product_id="cake00ka002034",
                quantity=1,
                icing_text=icing_text,
            )
        ],
        recipient=Recipient(name="Ada Lovelace", phone="0771234567"),
        delivery=Delivery(
            address="123 Galle Road",
            city="Colombo 03",
            date="2026-06-10",
        ),
        sender=Sender(name="Charles Babbage"),
        gift_message=gift_message,
    )


def test_create_order_gift_message_max_300_chars() -> None:
    """gift_message accepts exactly 300 characters."""
    message = "x" * 300
    order = _minimal_create_order_input(gift_message=message)
    assert order.gift_message == message


def test_create_order_gift_message_rejects_over_300_chars() -> None:
    """gift_message over 300 characters fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        _minimal_create_order_input(gift_message="x" * 301)

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("gift_message",) for error in errors)


def test_cart_item_icing_text_max_120_chars() -> None:
    """icing_text accepts exactly 120 characters."""
    text = "y" * 120
    item = CartItem(product_id="cake00ka002034", icing_text=text)
    assert item.icing_text == text


def test_cart_item_icing_text_rejects_over_120_chars() -> None:
    """icing_text over 120 characters fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        CartItem(product_id="cake00ka002034", icing_text="z" * 121)

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("icing_text",) for error in errors)


def test_create_order_cart_size_bounds() -> None:
    """Cart must contain 1–30 items."""
    with pytest.raises(ValidationError):
        CreateOrderInput(
            cart=[],
            recipient=Recipient(name="A", phone="0771234567"),
            delivery=Delivery(address="123 Main St", city="Colombo 03", date="2026-06-10"),
            sender=Sender(name="B"),
        )

    items = [CartItem(product_id=f"prod{i:04d}", quantity=1) for i in range(30)]
    order = CreateOrderInput(
        cart=items,
        recipient=Recipient(name="A", phone="0771234567"),
        delivery=Delivery(address="123 Main St", city="Colombo 03", date="2026-06-10"),
        sender=Sender(name="B"),
    )
    assert len(order.cart) == 30
