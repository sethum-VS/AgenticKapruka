"""Unit tests for lib.chat.product_honesty artificial floral disclosures."""

from __future__ import annotations

from lib.chat.product_honesty import (
    artificial_floral_note_for_picks,
    disclaimer_for_product,
    is_artificial_floral,
    is_flowers_request,
)


def _product(name: str, *, summary: str = "") -> dict[str, str]:
    return {"name": name, "summary": summary}


def test_is_artificial_floral_detects_silk_rose_product() -> None:
    product = _product(
        "Kit Kat Silk Roses Bouquet",
        summary="Kitkat, Roses, Valentine, Chocolates, Sweetbuds",
    )
    assert is_artificial_floral(product) is True


def test_is_artificial_floral_fresh_rose_no_disclaimer() -> None:
    product = _product(
        "6 Red Rose Bouquet With Elegant Wrapping",
        summary="flowers - Bouquet, Redroses fresh cut roses",
    )
    assert is_artificial_floral(product) is False
    assert disclaimer_for_product(product) is None


def test_is_artificial_floral_detects_soap_flower() -> None:
    product = _product(
        "Handmade Soap Flower Bouquet",
        summary="Decorative soap roses arranged as a bouquet",
    )
    assert is_artificial_floral(product) is True
    disclaimer = disclaimer_for_product(product)
    assert disclaimer is not None
    assert "soap flower bouquet" in disclaimer.lower()
    assert "not fresh-cut" in disclaimer.lower()


def test_is_artificial_floral_detects_paper_flower() -> None:
    product = _product("Paper Flower Arrangement", summary="Craft paper roses")
    assert is_artificial_floral(product) is True


def test_artificial_floral_note_for_picks_on_flowers_request() -> None:
    silk = _product("Kit Kat Silk Roses Bouquet")
    note = artificial_floral_note_for_picks(
        [silk],
        user_message="chocolate and flowers wife birthday",
    )
    assert note is not None
    assert "artificial" in note.lower()
    assert "not fresh-cut" in note.lower()


def test_artificial_floral_note_for_picks_on_fresh_flowers_request() -> None:
    silk = _product("Kit Kat Silk Roses Bouquet")
    note = artificial_floral_note_for_picks(
        [silk],
        user_message="I need fresh flowers for an anniversary",
    )
    assert note is not None
    assert "artificial" in note.lower()
    assert "not fresh-cut" in note.lower()


def test_artificial_floral_note_skipped_without_flowers_request() -> None:
    silk = _product("Kit Kat Silk Roses Bouquet")
    assert (
        artificial_floral_note_for_picks(
            [silk],
            user_message="chocolate gift for wife birthday",
        )
        is None
    )


def test_is_flowers_request_matches_roses_and_bouquet() -> None:
    assert is_flowers_request("show me flowers for a birthday")
    assert is_flowers_request("red roses bouquet")
    assert not is_flowers_request("chocolate gift hamper")
