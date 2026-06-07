"""Snapshot and structure tests for templates/components/product_card.html."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.templating import _create_templates, normalize_html_snapshot, render_product_card

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
PRODUCTS_DIR = FIXTURES_DIR / "products"
SNAPSHOTS_DIR = FIXTURES_DIR / "snapshots"


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def _load_product_fixture(name: str) -> dict[str, object]:
    return json.loads((PRODUCTS_DIR / name).read_text(encoding="utf-8"))


def _load_snapshot(name: str) -> str:
    return (SNAPSHOTS_DIR / name).read_text(encoding="utf-8")


def test_product_card_snapshot_in_stock() -> None:
    """Rendered HTML matches golden snapshot for sample in-stock product fixture."""
    product = _load_product_fixture("sample_in_stock.json")
    html = normalize_html_snapshot(render_product_card(product))
    expected = normalize_html_snapshot(_load_snapshot("product_card_in_stock.html"))

    assert html == expected


def test_product_card_renders_required_fields() -> None:
    """Card exposes id, name, formatted price, image, stock badge, and add-to-cart HTMX."""
    product = _load_product_fixture("sample_in_stock.json")
    html = render_product_card(product)

    assert 'data-product-id="cake00ka002034"' in html
    assert "Chocolate Fudge Birthday Cake" in html
    assert "LKR 4,500" in html
    assert 'src="https://cdn.kapruka.com/cakes/chocolate-fudge.jpg"' in html
    assert 'data-testid="stock-badge"' in html
    assert "In Stock" in html
    assert 'hx-post="/cart/add"' in html
    assert 'hx-vals=\'{"product_id": "cake00ka002034"}\'' in html
    assert 'hx-target="#cart-panel"' in html
    assert 'hx-swap="outerHTML"' in html
    assert "Add to cart" in html
    assert "hover:shadow-md" in html
    assert 'href="https://www.kapruka.com/cakes/chocolate-fudge"' in html


def test_product_card_out_of_stock_disables_add_button() -> None:
    """Out-of-stock products show overlay badge and disable the cart button."""
    product = _load_product_fixture("sample_out_of_stock.json")
    html = render_product_card(product)

    assert 'data-testid="stock-overlay"' in html
    assert "Out of Stock" in html
    assert "disabled" in html
    assert 'aria-disabled="true"' in html
    assert "LKR 3,200" in html
