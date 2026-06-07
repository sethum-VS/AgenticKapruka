"""Structure tests for templates/components/stock_badge.html."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.templating import _create_templates, render_product_card, render_stock_badge

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
PRODUCTS_DIR = FIXTURES_DIR / "products"


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def _load_product_fixture(name: str) -> dict[str, object]:
    return json.loads((PRODUCTS_DIR / name).read_text(encoding="utf-8"))


def test_stock_badge_in_stock_renders_green_badge() -> None:
    """In-stock products show green badge with stock level metadata."""
    html = render_stock_badge(in_stock=True, stock_level="high")

    assert 'data-testid="stock-badge"' in html
    assert "In Stock" in html
    assert "bg-emerald-600/90" in html
    assert 'data-stock-level="high"' in html
    assert 'data-testid="stock-overlay"' not in html


def test_stock_badge_low_stock_renders_amber_badge() -> None:
    """Low stock_level still in stock shows amber Low Stock badge."""
    html = render_stock_badge(in_stock=True, stock_level="low")

    assert 'data-testid="stock-badge"' in html
    assert "Low Stock" in html
    assert "bg-amber-500/90" in html
    assert 'data-stock-level="low"' in html


def test_stock_badge_out_of_stock_renders_muted_overlay() -> None:
    """Out-of-stock products show full-image muted overlay."""
    html = render_stock_badge(in_stock=False, stock_level="out")

    assert 'data-testid="stock-overlay"' in html
    assert "Out of Stock" in html
    assert "bg-commerce-ink/40" in html
    assert "bg-commerce-muted/90" in html
    assert 'data-stock-level="out"' in html
    assert 'data-testid="stock-badge"' not in html


def test_product_card_includes_stock_badge_partial_in_stock() -> None:
    """Product card embeds stock_badge partial for in-stock fixture."""
    product = _load_product_fixture("sample_in_stock.json")
    html = render_product_card(product)

    assert 'data-testid="stock-badge"' in html
    assert "In Stock" in html
    assert 'data-stock-level="high"' in html


def test_product_card_includes_stock_badge_partial_out_of_stock() -> None:
    """Product card embeds stock_badge partial for out-of-stock fixture."""
    product = _load_product_fixture("sample_out_of_stock.json")
    html = render_product_card(product)

    assert 'data-testid="stock-overlay"' in html
    assert "Out of Stock" in html
    assert 'data-stock-level="out"' in html
