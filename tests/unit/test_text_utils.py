"""Tests for lib.utils.text helpers."""

from __future__ import annotations

from lib.kapruka.types import CategoryRef, Money, ProductResult
from lib.utils.text import decode_html_entities


def test_decode_html_entities_decodes_en_dash() -> None:
    assert decode_html_entities("Cadbury 135g &#8211; 30 Minis") == "Cadbury 135g – 30 Minis"


def test_decode_html_entities_repairs_mangled_catalog_pattern() -> None:
    raw = "Cadbury Milk Chocolate Chunks 135g N#226;n#8364;n#8220; 30 Minis"
    decoded = decode_html_entities(raw)
    assert "N#" not in decoded
    assert "–" in decoded or "€" in decoded or '"' in decoded


def test_decode_html_entities_is_idempotent() -> None:
    value = "Plain chocolate"
    assert decode_html_entities(value) == value


def test_product_result_decodes_html_entities_in_name() -> None:
    product = ProductResult(
        id="choc001",
        name="Cadbury Milk Chocolate Chunks 135g &#8211; 30 Minis",
        summary="Mini chocolate chunks.",
        price=Money(amount=1200.0, currency="LKR"),
        compare_at_price=None,
        in_stock=True,
        stock_level="high",
        image_url=None,
        category=CategoryRef(id="cat_choc", name="Chocolate", slug="chocolate"),
        rating=None,
        ships_internationally=False,
        url="https://www.kapruka.com/example",
    )
    assert product.name == "Cadbury Milk Chocolate Chunks 135g – 30 Minis"
