"""Tests for Jinja2 template environment."""

from __future__ import annotations

import pytest
from starlette.requests import Request

from app.templating import (
    SUPPORTED_CURRENCY_CODES,
    _create_templates,
    format_currency,
    get_templates,
)
from lib.chat.page_context import cart_template_context, currency_template_context


def _make_request() -> Request:
    scope: dict[str, object] = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "query_string": b"",
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_format_currency_filter_registered() -> None:
    assert format_currency(1500, "LKR") == "Rs. 1,500"


def test_get_templates_returns_singleton() -> None:
    first = get_templates()
    second = get_templates()
    assert first is second
    assert "format_currency" in first.env.filters


def test_template_response_renders_base_html() -> None:
    """TemplateResponse renders base.html with HTMX, Alpine.js, and SSE extension."""
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

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AgenticKapruka" in html
    assert "/static/js/vendor/htmx.min.js" in html
    assert "/static/js/vendor/htmx-ext-sse.min.js" in html
    assert "/static/js/vendor/alpine.min.js" in html
    assert 'defer src="/static/js/vendor/htmx.min.js"' in html
    assert "icon_names=add,add_comment,history,local_shipping,menu,redeem,send,shopping_cart" in html
    assert 'hx-ext="sse"' in html
    assert 'href="/static/css/app.css"' in html
    assert "/static/js/cart-drawer.js" in html
    assert 'data-testid="cart-drawer"' in html
    assert 'data-testid="header-currency"' in html
    assert 'hx-post="/session/currency"' in html
    assert len(SUPPORTED_CURRENCY_CODES) == 6
