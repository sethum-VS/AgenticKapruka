"""Unit tests for graphs.nodes.resolve_cart_product."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.resolve_cart_product import (
    match_products_by_phrase,
    phrase_product_overlap_score,
    resolve_cart_product,
)
from graphs.nodes.execute_cart_action import execute_cart_action
from graphs.state import AgentState
from lib.kapruka.errors import KaprukaValidationError
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


_AMMAS_DELIGHT = {
    "id": "cake00amma001",
    "name": "Ammas Delightful Creation",
    "summary": "A delightful cake.",
    "price": {"amount": 7500.0, "currency": "LKR"},
    "in_stock": True,
    "stock_level": "high",
    "image_url": None,
    "category": {"id": "cat_cake", "name": "Cake", "slug": "cake"},
    "ships_internationally": False,
    "url": "https://example.com/amma",
}


def test_match_products_by_phrase_normalizes_possessive_apostrophe() -> None:
    product, tied, question = match_products_by_phrase(
        "Amma's Delightful Creation",
        [_AMMAS_DELIGHT],
    )
    assert product is not None
    assert product["id"] == "cake00amma001"
    assert not tied
    assert question is None


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
async def test_resolve_cart_product_deictic_that_skips_cold_mcp() -> None:
    mock_service = AsyncMock()
    state: AgentState = {
        "messages": [HumanMessage(content="Add that to my cart")],
        "session_id": "sess-cart-that",
        "last_visible_products": [_BLUSH_ROSES],
        "last_search_products": [_BLUSH_ROSES, _RED_ROSES],
    }

    result = await resolve_cart_product(
        state,
        kapruka_service=mock_service,
        client_ip="127.0.0.1",
    )

    action = result["cart_action_result"]
    assert action["status"] == "resolved"
    assert action["product"]["id"] == "combo00blush001"
    mock_service.search_products.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_cart_product_ordinal_first_skips_cold_mcp() -> None:
    mock_service = AsyncMock()
    state: AgentState = {
        "messages": [HumanMessage(content="Add the first one to my cart")],
        "session_id": "sess-cart-ordinal",
        "last_visible_products": [_RED_ROSES, _BLUSH_ROSES],
    }

    result = await resolve_cart_product(state, kapruka_service=mock_service)

    action = result["cart_action_result"]
    assert action["status"] == "resolved"
    assert action["product"]["id"] == "combo00red001"
    mock_service.search_products.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_execute_cart_action_softens_live_stock_mismatch() -> None:
    from unittest.mock import patch

    from lib.redis.cart import StoredCartItem

    mock_service = AsyncMock()
    mock_service.get_product.side_effect = KaprukaValidationError(
        code="product_out_of_stock",
        message="Out of stock",
    )
    mock_redis = MagicMock()
    stored_item = StoredCartItem(
        product_id="cake00amma001",
        quantity=1,
        name="Ammas Delightful Creation",
        price_amount=7500.0,
        price_currency="LKR",
    )

    state: AgentState = {
        "session_id": "sess-cart-stock-mismatch",
        "currency": "LKR",
        "cart_action_result": {
            "status": "resolved",
            "product": {
                "id": "cake00amma001",
                "name": "Ammas Delightful Creation",
                "price": {"amount": 7500.0, "currency": "LKR"},
                "in_stock": True,
            },
        },
    }

    with (
        patch(
            "graphs.nodes.execute_cart_action.get_cart",
            new_callable=AsyncMock,
        ) as mock_get_cart,
        patch(
            "graphs.nodes.execute_cart_action.add_item",
            new_callable=AsyncMock,
        ) as mock_add_item,
    ):
        mock_get_cart.return_value = []
        mock_add_item.return_value = stored_item
        mock_get_cart.side_effect = [[], [stored_item]]

        result = await execute_cart_action(
            state,
            redis_client=mock_redis,
            kapruka_service=mock_service,
            client_ip="127.0.0.1",
        )

    action = result["cart_action_result"]
    assert action["status"] == "added"
    assert "stock_warning" in action
    mock_add_item.assert_awaited_once()
