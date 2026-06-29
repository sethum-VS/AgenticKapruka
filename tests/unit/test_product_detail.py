"""Unit tests for lib.chat.product_detail."""

from __future__ import annotations

import pytest

from lib.chat.product_detail import (
    enrich_tool_results_with_session_product,
    is_product_detail_turn,
    is_sweetness_preference_turn,
    match_product_from_last_search,
    merge_with_session_resolved,
    normalize_resolved_product,
    product_preference_note,
    product_weight,
    resolve_product_detail,
    summarize_product_from_carousel,
)
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL


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


def test_summarize_product_from_carousel_includes_id_and_weight() -> None:
    product = {
        "id": "CAKE00KA001685",
        "name": "Springtime Birthday Ribbon Cake",
        "summary": "Fresh sponge with ribbon.",
        "price": {"amount": 5770.0, "currency": "LKR"},
        "attributes": {"weight": "2.77"},
    }
    summary = summarize_product_from_carousel(product)
    assert "CAKE00KA001685" in summary
    assert "2.77 Lbs" in summary
    assert "Rs. 5,770" in summary


def test_resolve_product_detail_uses_session_when_carousel_lacks_weight() -> None:
    carousel = {
        "id": "CAKE00KA001685",
        "name": "Springtime Birthday Ribbon Cake",
        "summary": "Fresh sponge.",
        "price": {"amount": 5770.0, "currency": "LKR"},
    }
    session = normalize_resolved_product(
        {
            "id": "CAKE00KA001685",
            "name": "Springtime Birthday Ribbon Cake",
            "attributes": {"weight": "2.77"},
            "price": {"amount": 5770.0, "currency": "LKR"},
        },
    )
    product, session_update = resolve_product_detail(
        get_payload=None,
        matched=carousel,
        session_resolved=session,
    )
    assert product is not None
    assert product_weight(product) == "2.77"
    assert session_update is session


def test_merge_with_session_resolved_skips_mismatched_ids() -> None:
    carousel = {"id": "cake-a", "name": "Cake A"}
    session = {"id": "cake-b", "name": "Cake B", "attributes": {"weight": "1kg"}}
    merged = merge_with_session_resolved(carousel, session)
    assert merged == carousel


def test_enrich_tool_results_with_session_product_injects_persisted_detail() -> None:
    session = normalize_resolved_product(
        {
            "id": "CAKE00KA001685",
            "name": "Springtime Birthday Ribbon Cake",
            "attributes": {"weight": "2.77"},
        },
    )
    enriched = enrich_tool_results_with_session_product(
        {},
        session,
        product_id="CAKE00KA001685",
        get_product_tool=GET_PRODUCT_TOOL,
    )
    assert enriched is not None
    assert enriched[GET_PRODUCT_TOOL]["attributes"]["weight"] == "2.77"


def test_product_preference_note_less_sweet_honest_default() -> None:
    product = {
        "name": "Springtime Birthday Ribbon Cake",
        "summary": "Pastel ribbon cake for birthdays.",
    }
    note = product_preference_note("is it suitable for someone who prefers less sweet cakes?", product)
    assert note is not None
    assert "does not list exact sweetness" in note.lower()


def test_summarize_product_from_carousel_includes_preference_note() -> None:
    product = {
        "id": "CAKE00KA001685",
        "name": "Springtime Birthday Ribbon Cake",
        "summary": "Pastel ribbon cake.",
        "price": {"amount": 5770.0, "currency": "LKR"},
        "attributes": {"weight": "2.77"},
    }
    summary = summarize_product_from_carousel(
        product,
        user_message="tell me more — is it less sweet?",
    )
    assert "2.77" in summary
    assert "sweetness" in summary.lower()


@pytest.mark.parametrize(
    "message",
    [
        "How much does the Springtime Birthday Ribbon Cake weigh?",
        "The Springtime one looks nice. How much does it weigh, and is it less sweet?",
        "what is the weight of that cake?",
    ],
)
def test_is_product_detail_turn_matches_weight_phrasing(message: str) -> None:
    assert is_product_detail_turn(message)


def test_is_sweetness_preference_turn() -> None:
    assert is_sweetness_preference_turn("elegant, not too sweet")
    assert not is_sweetness_preference_turn("birthday cake for mom")


def test_match_product_from_last_search_name_mention_in_long_detail_question() -> None:
    """Long weight+sweetness follow-ups still resolve the named carousel item."""
    carousel = [
        {
            "id": "CAKE00KA001685",
            "name": "Springtime Birthday Ribbon Cake",
            "summary": "Pastel ribbon cake.",
        },
        {
            "id": "CAKE00KA001827",
            "name": "Happy Birthday Symphony Ribbon Cake",
            "summary": "Celebration centerpiece.",
        },
    ]
    message = (
        "The Springtime one looks nice. "
        "How much does it weigh, and is it less sweet than typical birthday cakes?"
    )
    matched = match_product_from_last_search(
        message,
        carousel,
        last_visible_products=carousel,
    )
    assert matched is not None
    assert matched["id"] == "CAKE00KA001685"
