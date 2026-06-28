"""Unit tests for lib.chat.product_detail."""

from __future__ import annotations

from lib.chat.product_detail import (
    enrich_tool_results_with_session_product,
    match_product_from_last_search,
    merge_with_session_resolved,
    normalize_resolved_product,
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
