"""Isolation tests for templates/chat/message_assistant.html."""

from __future__ import annotations

import pytest

from app.templating import _create_templates
from graphs.nodes.generate_response import render_assistant_html

_SAMPLE_PRODUCTS_HTML = (
    '<div class="flex gap-3 overflow-x-auto" data-testid="product-carousel-stub">'
    '<article class="shrink-0 w-40">Chocolate Cake</article>'
    '<article class="shrink-0 w-40">Vanilla Cake</article>'
    "</div>"
)


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_message_assistant_renders_message_only_structure() -> None:
    """Partial renders left-aligned assistant bubble with prose styling."""
    html = render_assistant_html("Hello from Kapruka!")

    assert "Hello from Kapruka!" in html
    assert 'data-role="assistant-message"' in html
    assert 'role="assistant"' in html
    assert 'aria-label="Assistant message"' in html
    assert "justify-start" in html
    assert "prose-assistant" in html
    assert 'data-slot="product-carousel"' not in html


def test_message_assistant_renders_optional_products_block() -> None:
    """Optional products_html slot accepts pre-rendered carousel markup."""
    html = render_assistant_html(
        "Here are some birthday cakes for you.",
        products_html=_SAMPLE_PRODUCTS_HTML,
    )

    assert "Here are some birthday cakes for you." in html
    assert 'data-slot="product-carousel"' in html
    assert 'aria-label="Suggested products"' in html
    assert 'data-testid="product-carousel-stub"' in html
    assert "Chocolate Cake" in html
    assert "Vanilla Cake" in html
    assert "assistant-products" in html


def test_message_assistant_escapes_message_text() -> None:
    """User-visible message text is HTML-escaped by Jinja autoescape."""
    html = render_assistant_html('<script>alert("xss")</script>')

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_message_assistant_products_html_is_not_escaped() -> None:
    """Pre-rendered carousel HTML is injected via safe filter for HTMX partial swap."""
    templates = _create_templates()
    template = templates.env.get_template("chat/message_assistant.html")
    html = template.render(
        message="Browse these picks.",
        products_html='<div data-carousel="1"></div>',
    )

    assert '<div data-carousel="1"></div>' in html
