"""Unit tests for graphs.nodes.resolve_cart_product."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.resolve_cart_product import (
    match_products_by_phrase,
    phrase_product_overlap_score,
    resolve_cart_product,
)
from graphs.state import AgentState
from lib.kapruka.types import (
    CategoryRef,
    Money,
    ProductResult,
    SearchProductsOutput,
)

_BLUSH_ROSES = {
    "id": "combo00blush001",
    "name": "Blush Roses Combo Gift",
    "summary": "Roses and chocolates.",
    "price": {"amount": 6500.0, "currency": "LKR"},
    "in_stock": True,
    "stock_level": "high",
    "image_url": None,
    "category": {"id": "cat_combo", "name": "Combo", "slug": "combo"},
    "ships_internationally": False,
    "url": "https://example.com/blush",
}

_RED_ROSES = {
    **_BLUSH_ROSES,
    "id": "combo00red001",
    "name": "Red Roses Combo Gift",
}


def test_phrase_product_overlap_score_prefers_matching_tokens() -> None:
    assert phrase_product_overlap_score("Blush Roses combo", "Blush Roses Combo Gift") >= 0.6
    assert phrase_product_overlap_score("sunflower bouquet", "Blush Roses Combo Gift") < 0.6


def test_match_products_by_phrase_single_winner() -> None:
    product, tied, question = match_products_by_phrase(
        "Blush Roses combo",
        [_BLUSH_ROSES, _RED_ROSES],
    )
    assert product is not None
    assert product["id"] == "combo00blush001"
    assert not tied
    assert question is None


def test_match_products_by_phrase_tie_returns_clarifying_question() -> None:
    twin_a = {**_BLUSH_ROSES, "id": "a", "name": "Blush Roses Deluxe Combo"}
    twin_b = {**_BLUSH_ROSES, "id": "b", "name": "Blush Roses Premium Combo"}
    product, tied, question = match_products_by_phrase("Blush Roses combo", [twin_a, twin_b])
    assert product is None
    assert len(tied) == 2
    assert question is not None
    assert "Which one" in question


@pytest.mark.asyncio
async def test_resolve_cart_product_uses_last_search_products() -> None:
    state: AgentState = {
        "messages": [
            HumanMessage(content="Add the Blush Roses combo to my cart please"),
        ],
        "session_id": "sess-cart-resolve",
        "last_search_products": [_BLUSH_ROSES, _RED_ROSES],
    }

    result = await resolve_cart_product(state)

    action = result["cart_action_result"]
    assert action["status"] == "resolved"
    assert action["product"]["id"] == "combo00blush001"


@pytest.mark.asyncio
async def test_resolve_cart_product_cold_start_searches_mcp() -> None:
    blush = ProductResult(
        id="combo00blush001",
        name="Blush Roses Combo Gift",
        summary="Roses and chocolates.",
        price=Money(amount=6500.0, currency="LKR"),
        compare_at_price=None,
        in_stock=True,
        stock_level="high",
        image_url=None,
        category=CategoryRef(id="cat_combo", name="Combo", slug="combo"),
        rating=None,
        ships_internationally=False,
        url="https://example.com/blush",
    )
    mock_service = AsyncMock()
    mock_service.search_products.return_value = SearchProductsOutput(
        results=[blush],
        next_cursor=None,
        applied_filters={"q": "Blush Roses combo", "limit": 10, "in_stock_only": False},
    )

    state: AgentState = {
        "messages": [
            HumanMessage(content="Add the Blush Roses combo to my cart please"),
        ],
        "session_id": "sess-cart-cold",
        "currency": "LKR",
    }

    result = await resolve_cart_product(
        state,
        kapruka_service=mock_service,
        client_ip="127.0.0.1",
    )

    action = result["cart_action_result"]
    assert action["status"] == "resolved"
    assert action["product"]["id"] == "combo00blush001"
    mock_service.search_products.assert_awaited_once()
