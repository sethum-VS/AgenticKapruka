"""Unit tests for lib.chat.product_reference."""

from __future__ import annotations

from lib.chat.product_reference import (
    is_deictic_phrase,
    is_ordinal_phrase,
    resolve_product_reference,
)

_BLUSH = {"id": "a", "name": "Blush Roses Combo Gift"}
_RED = {"id": "b", "name": "Red Roses Combo Gift"}
_THIRD = {"id": "c", "name": "Sunflower Bouquet"}


def test_is_deictic_phrase() -> None:
    assert is_deictic_phrase("that")
    assert is_deictic_phrase("This one")
    assert not is_deictic_phrase("Blush Roses combo")


def test_is_ordinal_phrase() -> None:
    assert is_ordinal_phrase("first")
    assert is_ordinal_phrase("the second one")
    assert is_ordinal_phrase("3rd")
    assert not is_ordinal_phrase("chocolate gift")


def test_resolve_ordinal_with_trailing_descriptor() -> None:
    result = resolve_product_reference(
        "the first flower bouquet",
        last_visible_products=[_BLUSH, _RED],
        last_search_products=[_BLUSH, _RED],
    )
    assert result is not None
    assert result["status"] == "resolved"
    assert result["product"]["id"] == "a"


def test_resolve_deictic_single_product() -> None:
    result = resolve_product_reference(
        "that",
        last_visible_products=[_BLUSH],
        last_search_products=[_BLUSH, _RED],
    )
    assert result is not None
    assert result["status"] == "resolved"
    assert result["product"]["id"] == "a"


def test_resolve_deictic_multi_clarify() -> None:
    result = resolve_product_reference(
        "that",
        last_visible_products=[_BLUSH, _RED],
        last_search_products=[_BLUSH, _RED],
    )
    assert result is not None
    assert result["status"] == "clarify"
    assert result["clarifying_question"] is not None
    assert "1)" in result["clarifying_question"]
    assert "2)" in result["clarifying_question"]


def test_resolve_ordinal_uses_visible_products() -> None:
    result = resolve_product_reference(
        "second",
        last_visible_products=[_BLUSH, _RED],
        last_search_products=[_THIRD],
    )
    assert result is not None
    assert result["status"] == "resolved"
    assert result["product"]["id"] == "b"


def test_resolve_deictic_empty_context() -> None:
    result = resolve_product_reference(
        "that",
        last_visible_products=None,
        last_search_products=None,
    )
    assert result is not None
    assert result["status"] == "clarify"
    assert "Search for a gift first" in (result["clarifying_question"] or "")
