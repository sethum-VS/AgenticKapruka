"""Unit tests for cart add failure HTMX responses."""

from __future__ import annotations

import pytest

from app.templating import _create_templates, render_cart_add_error_response
from lib.redis.cart import StoredCartItem


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_cart_add_error_response_includes_banner_retry_and_panel() -> None:
    html = render_cart_add_error_response(
        items=[],
        product_id="cake00ka002034",
        error_code="upstream_error",
        message="Please try again in a moment.",
        quantity=1,
    )

    assert 'id="cart-panel"' in html
    assert 'data-testid="cart-add-error"' in html
    assert 'data-testid="error-banner"' in html
    assert 'data-error-code="upstream_error"' in html
    assert "Please try again in a moment." in html
    assert 'data-testid="cart-add-retry"' in html
    assert 'hx-post="/cart/add"' in html
    assert '"product_id": "cake00ka002034"' in html


def test_cart_add_error_response_preserves_existing_cart_items() -> None:
    items = [
        StoredCartItem(
            product_id="cake00ka002034",
            quantity=1,
            name="Chocolate Cake",
            price_amount=4500.0,
            price_currency="LKR",
        ),
    ]
    html = render_cart_add_error_response(
        items=items,
        product_id="cake00ka001827",
        error_code="upstream_error",
        message="Kapruka is temporarily unavailable.",
    )

    assert "Chocolate Cake" in html
    assert 'data-testid="cart-line-item"' in html
