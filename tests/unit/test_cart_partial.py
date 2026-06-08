"""Unit tests for cart partial template rendering."""

from __future__ import annotations

import pytest

from app.templating import _create_templates, render_cart_partial
from lib.redis.cart import StoredCartItem


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_cart_partial_empty_state() -> None:
    html = render_cart_partial(items=[])

    assert 'id="cart-panel"' in html
    assert 'data-testid="cart-panel"' in html
    assert 'data-item-count="0"' in html
    assert 'data-testid="cart-empty"' in html
    assert "Your cart is empty." in html


def test_cart_partial_lists_item_with_stepper_and_remove() -> None:
    items = [
        StoredCartItem(
            product_id="cake00ka002034",
            quantity=2,
            name="Chocolate Fudge Birthday Cake",
            price_amount=4500.0,
            price_currency="LKR",
        ),
    ]
    html = render_cart_partial(items=items)

    assert "Chocolate Fudge Birthday Cake" in html
    assert 'data-testid="cart-line-item"' in html
    assert 'hx-post="/cart/remove"' in html
    assert 'hx-post="/cart/update"' in html
    assert 'hx-target="#cart-panel"' in html
    assert 'hx-swap="outerHTML"' in html
    assert 'data-testid="cart-quantity-stepper"' in html
    assert 'data-testid="cart-qty-value"' in html
    assert ">2<" in html.replace("\n", "").replace(" ", "")
    assert "Rs. 9,000" in html
    assert 'data-item-count="2"' in html
    assert 'data-testid="cart-proceed-checkout"' in html


def test_cart_partial_disables_decrease_at_quantity_one() -> None:
    items = [
        StoredCartItem(
            product_id="cake00ka002034",
            quantity=1,
            name="Vanilla Cake",
            price_amount=3000.0,
            price_currency="LKR",
        ),
    ]
    html = render_cart_partial(items=items)

    assert 'data-testid="cart-qty-decrease"' in html
    assert 'disabled aria-disabled="true"' in html
    assert '"quantity": 2' in html
