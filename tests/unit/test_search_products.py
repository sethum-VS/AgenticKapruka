"""Unit tests for kapruka_search_products wrapper."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lib.kapruka.errors import KaprukaError
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.tools.search_products import TOOL_NAME, search_products

_SAMPLE_JSON: dict[str, Any] = {
    "results": [
        {
            "id": "cake00ka002034",
            "name": "Chocolate Birthday Cake",
            "summary": "Rich chocolate cake for celebrations.",
            "price": {"amount": 4500.0, "currency": "LKR"},
            "compare_at_price": None,
            "in_stock": True,
            "stock_level": "high",
            "image_url": "https://static2.kapruka.com/product-image/cake.jpg",
            "category": {
                "id": "cat_cakes",
                "name": "Birthday",
                "slug": "birthday",
            },
            "rating": None,
            "ships_internationally": False,
            "url": "https://www.kapruka.com/buyonline/chocolate-birthday-cake/kid/cake00ka002034",
        }
    ],
    "next_cursor": "eyJ1IjoiTXc9PSIsInAiOjF9",
    "applied_filters": {
        "q": "birthday cake",
        "limit": 10,
        "in_stock_only": False,
    },
}


@pytest.fixture
def mcp_client() -> MCPHttpClient:
    client = AsyncMock(spec=MCPHttpClient)
    client.call_tool = AsyncMock(return_value=json.dumps(_SAMPLE_JSON))
    return client


async def test_search_products_parses_mocked_json(mcp_client: MCPHttpClient) -> None:
    """Mocked MCP JSON is parsed into a typed ProductResult list."""
    result = await search_products(
        mcp_client,
        q="birthday cake",
        category="Birthday",
        min_price=1000.0,
        max_price=10000.0,
        in_stock_only=True,
        sort="price_asc",
        limit=5,
        cursor="page-2",
        currency="USD",
    )

    assert len(result.results) == 1
    product = result.results[0]
    assert product.id == "cake00ka002034"
    assert product.name == "Chocolate Birthday Cake"
    assert product.price.amount == 4500.0
    assert product.price.currency == "LKR"
    assert product.in_stock is True
    assert product.category.name == "Birthday"
    assert result.next_cursor == "eyJ1IjoiTXc9PSIsInAiOjF9"
    assert result.applied_filters["q"] == "birthday cake"


async def test_search_products_forces_response_format_json(
    mcp_client: MCPHttpClient,
) -> None:
    """MCP params always include response_format=json."""
    await search_products(mcp_client, q="roses")

    mcp_client.call_tool.assert_awaited_once_with(  # type: ignore[attr-defined]
        TOOL_NAME,
        {
            "q": "roses",
            "limit": 10,
            "currency": "LKR",
            "in_stock_only": False,
            "sort": "relevance",
            "include_stubs": False,
            "response_format": "json",
        },
    )


async def test_search_products_raises_on_no_products_message(
    mcp_client: MCPHttpClient,
) -> None:
    """Plain-text 'No products found' MCP responses raise KaprukaError."""
    mcp_client.call_tool = AsyncMock(  # type: ignore[method-assign]
        return_value="No products found for 'xyz'"
    )

    with pytest.raises(KaprukaError) as exc_info:
        await search_products(mcp_client, q="xyz")

    assert exc_info.value.code == "no_products_found"


async def test_search_products_validates_query_before_mcp_call(
    mcp_client: MCPHttpClient,
) -> None:
    """SearchProductsInput rejects queries shorter than 3 characters."""
    with pytest.raises(ValueError, match="at least 3"):
        await search_products(mcp_client, q="ab")

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]
