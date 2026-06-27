"""Unit tests for lib.chat.product_detail."""

from __future__ import annotations

from lib.chat.product_detail import match_product_from_last_search


def test_match_product_from_last_search_ordinal_first_cake() -> None:
    visible = [
        {"id": "cake-001", "name": "Springtime Celebration Cake", "summary": "Fresh sponge."},
        {"id": "cake-002", "name": "Chocolate Delight", "summary": "Rich cocoa."},
    ]
    matched = match_product_from_last_search(
        "Tell me more about the first cake",
        last_search_products=visible,
        last_visible_products=visible,
        session_product_focus="cake",
    )
    assert matched is not None
    assert matched["id"] == "cake-001"
