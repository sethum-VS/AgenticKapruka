"""Unit tests for Alpine cart drawer component and template."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.requests import Request

from app.templating import _create_templates, render_cart_drawer, render_cart_partial
from lib.chat.page_context import cart_template_context, currency_template_context
from lib.redis.cart import StoredCartItem

CART_DRAWER_JS = Path(__file__).resolve().parent.parent.parent / "static" / "js" / "cart-drawer.js"


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def _make_request() -> Request:
    scope: dict[str, object] = {
        "type": "http",
        "method": "GET",
        "path": "/chat",
        "headers": [],
        "query_string": b"",
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_cart_drawer_js_registers_alpine_component() -> None:
    """cart-drawer.js defines cartDrawer with open/close and HTMX badge sync."""
    source = CART_DRAWER_JS.read_text()

    assert 'Alpine.data("cartDrawer"' in source
    assert "init()" in source
    assert "htmx:afterSwap" in source
    assert "openDrawer" in source
    assert 'htmx.ajax("GET", "/cart/panel"' in source
    assert "htmx:afterSettle" in source
    assert "syncCountFromPanel" in source
    assert 'target.id !== "cart-panel"' in source
    assert "data-item-count" in source


def test_cart_drawer_empty_state() -> None:
    html = render_cart_drawer(items=[])

    assert 'data-testid="cart-drawer"' in html
    assert 'x-data="cartDrawer(0)"' in html
    assert 'data-testid="cart-icon"' in html
    assert 'data-testid="cart-badge"' in html
    assert 'data-testid="cart-backdrop"' in html
    assert 'data-testid="cart-drawer-panel"' in html
    assert 'id="cart-panel"' in html
    assert 'data-testid="cart-empty"' in html
    assert "@keydown.escape.window" in html
    assert 'x-transition:enter-start="translate-x-full"' in html


def test_cart_drawer_shows_badge_count_from_items() -> None:
    items = [
        StoredCartItem(
            product_id="cake001",
            quantity=2,
            icing_text=None,
            name="Chocolate Cake",
            price_amount=4500.0,
            price_currency="LKR",
        ),
        StoredCartItem(
            product_id="flower001",
            quantity=1,
            icing_text=None,
            name="Roses",
            price_amount=3000.0,
            price_currency="LKR",
        ),
    ]
    html = render_cart_drawer(items=items)

    assert 'x-data="cartDrawer(3)"' in html
    assert 'data-item-count="3"' in html
    assert 'data-testid="cart-line-item"' in html
    assert 'aria-label="3 items in cart"' in html


def test_cart_template_context_sums_quantities() -> None:
    items = [
        StoredCartItem(
            product_id="cake001",
            quantity=2,
            icing_text=None,
            name="Cake",
            price_amount=100.0,
            price_currency="LKR",
        ),
    ]
    ctx = cart_template_context(items)

    assert ctx["cart_item_count"] == 2
    assert len(ctx["cart_items"]) == 1


def test_base_html_includes_cart_drawer() -> None:
    """base.html loads cart drawer script and renders header cart icon."""
    from app.templating import get_templates

    templates = get_templates()
    request = _make_request()
    response = templates.TemplateResponse(
        request,
        "base.html",
        {
            "title": "AgenticKapruka",
            **currency_template_context("LKR"),
            **cart_template_context([]),
        },
    )
    html = response.body.decode()

    assert "/static/js/cart-drawer.js" in html
    assert 'data-testid="cart-drawer"' in html
    assert 'data-testid="cart-icon"' in html


def test_cart_drawer_flex_scroll_containers_have_min_h_0() -> None:
    """Aside and inner scroll area need min-h-0 so line items scroll on desktop."""
    html = render_cart_drawer(items=[])

    assert 'data-testid="cart-drawer-panel"' in html
    assert "flex min-h-0 w-full max-w-sm flex-col" in html
    assert 'class="min-h-0 flex-1 overflow-y-auto bg-white px-4 py-4"' in html


def test_cart_panel_swap_target_inside_scroll_wrapper() -> None:
    """HTMX outerHTML on #cart-panel replaces only cart_partial, not the scroll shell."""
    items = [
        StoredCartItem(
            product_id="cake001",
            quantity=1,
            icing_text=None,
            name="Vanilla Cake",
            price_amount=3500.0,
            price_currency="LKR",
        ),
    ]
    drawer_html = render_cart_drawer(items=items)
    partial_html = render_cart_partial(items=items)

    scroll_idx = drawer_html.index("overflow-y-auto")
    panel_idx = drawer_html.index('\n  id="cart-panel"')
    assert scroll_idx < panel_idx
    assert '\n  id="cart-panel"' in partial_html
    assert "overflow-y-auto" not in partial_html


def test_cart_drawer_embeds_cart_partial_markup() -> None:
    items = [
        StoredCartItem(
            product_id="cake001",
            quantity=1,
            icing_text=None,
            name="Vanilla Cake",
            price_amount=3500.0,
            price_currency="LKR",
        ),
    ]
    drawer_html = render_cart_drawer(items=items)
    partial_html = render_cart_partial(items=items)

    assert 'id="cart-panel"' in drawer_html
    assert "Vanilla Cake" in drawer_html
    assert 'hx-target="#cart-panel"' in drawer_html
    assert 'id="cart-panel"' in partial_html
