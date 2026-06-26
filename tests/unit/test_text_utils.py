"""Tests for lib.utils.text helpers."""

from __future__ import annotations

from lib.kapruka.types import CategoryRef, Money, ProductResult
from lib.utils.text import (
    decode_html_entities,
    normalize_catalog_text,
    repair_utf8_mojibake,
)


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


_MOJIBAKE_EN_DASH = b"\xe2\x80\x93".decode("latin-1")


def test_repair_utf8_mojibake_fixes_en_dash() -> None:
    raw = f"Comfort And Sip Travel Gift Set {_MOJIBAKE_EN_DASH} Pink"
    repaired = repair_utf8_mojibake(raw)
    assert _MOJIBAKE_EN_DASH not in repaired
    assert "–" in repaired


def test_normalize_catalog_text_composes_mojibake_and_entities() -> None:
    raw = f"Comfort And Sip Travel Gift Set {_MOJIBAKE_EN_DASH} Pink"
    assert _MOJIBAKE_EN_DASH not in normalize_catalog_text(raw)
    assert "–" in normalize_catalog_text(raw)
    assert normalize_catalog_text("Plain &#8211; text") == "Plain – text"


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


def test_normalize_catalog_text_fixes_apostrophe_mojibake() -> None:
    raw = "Lindt Lindor â€™ Assorted"
    normalized = normalize_catalog_text(raw)
    assert "â€™" not in normalized
    assert "'" in normalized or "’" in normalized


def test_product_card_template_normalizes_mojibake_name() -> None:
    from app.templating import render_product_carousel

    product = {
        "id": "gift001",
        "name": f"Comfort Gift Set {_MOJIBAKE_EN_DASH} Pink",
        "price": {"amount": 4500.0, "currency": "LKR"},
        "in_stock": True,
        "stock_level": "high",
        "url": "https://www.kapruka.com/example",
        "image_url": None,
    }
    html = render_product_carousel([product])
    assert _MOJIBAKE_EN_DASH not in html
    assert "–" in html


def test_normalize_catalog_text_strips_wrapping_backticks() -> None:
    """`vibe Check` → vibe Check (backtick wrapping stripped)."""
    assert normalize_catalog_text("`vibe Check`") == "vibe Check"


def test_normalize_catalog_text_strips_triple_backticks() -> None:
    """```product name``` → product name."""
    assert normalize_catalog_text("```product name```") == "product name"


def test_normalize_catalog_text_collapses_double_space_after_strip() -> None:
    # Backticks stripped, then interior double-spaces collapsed to single
    result = normalize_catalog_text("`  hello  world  `")
    assert "hello" in result
    assert "  " not in result, f"Double spaces should be collapsed: {result!r}"
