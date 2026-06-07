"""Unit tests for kapruka_get_product wrapper."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lib.kapruka.errors import KaprukaNotFoundError
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.tools.get_product import TOOL_NAME, get_product

_SAMPLE_JSON: dict[str, Any] = {
    "id": "cake00ka002034",
    "name": "Chocolate Birthday Cake",
    "description": "Rich chocolate sponge with buttercream frosting.",
    "summary": "Perfect for birthday celebrations.",
    "price": {"amount": 4500.0, "currency": "LKR"},
    "compare_at_price": None,
    "in_stock": True,
    "stock_level": "high",
    "category": {
        "id": "cat_cakes",
        "name": "Birthday",
        "slug": "birthday",
        "path": "Cakes > Birthday",
    },
    "variants": [
        {
            "id": "var_default",
            "name": "Standard",
            "sku": "CAKE-CHOC-001",
            "price": {"amount": 4500.0, "currency": "LKR"},
            "in_stock": True,
            "stock_level": "high",
            "attributes": {"size": "1kg"},
        }
    ],
    "images": [
        "https://static2.kapruka.com/product-image/cake.jpg",
    ],
    "attributes": {
        "type": "cake",
        "subtype": "birthday",
        "weight": "1kg",
        "vendor": "Kapruka Bakery",
    },
    "shipping": {
        "ships_from": "Colombo",
        "ships_internationally": False,
        "restricted_countries": [],
    },
    "rating": None,
    "url": "https://www.kapruka.com/buyonline/chocolate-birthday-cake/kid/cake00ka002034",
}


@pytest.fixture
def mcp_client() -> MCPHttpClient:
    client = AsyncMock(spec=MCPHttpClient)
    client.call_tool = AsyncMock(return_value=json.dumps(_SAMPLE_JSON))
    return client


async def test_get_product_parses_mocked_json(mcp_client: MCPHttpClient) -> None:
    """Mocked MCP JSON is parsed into a typed GetProductOutput."""
    result = await get_product(
        mcp_client,
        product_id="cake00ka002034",
        currency="USD",
    )

    assert result.id == "cake00ka002034"
    assert result.name == "Chocolate Birthday Cake"
    assert result.price.amount == 4500.0
    assert result.price.currency == "LKR"
    assert result.in_stock is True
    assert result.category.name == "Birthday"
    assert result.category.path == "Cakes > Birthday"
    assert len(result.variants) == 1
    assert result.variants[0].sku == "CAKE-CHOC-001"
    assert result.attributes.type == "cake"
    assert result.shipping.ships_from == "Colombo"
    assert len(result.images) == 1


async def test_get_product_forces_response_format_json(
    mcp_client: MCPHttpClient,
) -> None:
    """MCP params always include response_format=json."""
    await get_product(mcp_client, product_id="cake00ka002034")

    mcp_client.call_tool.assert_awaited_once_with(  # type: ignore[attr-defined]
        TOOL_NAME,
        {
            "product_id": "cake00ka002034",
            "currency": "LKR",
            "response_format": "json",
        },
    )


async def test_get_product_raises_not_found_on_mcp_error(
    mcp_client: MCPHttpClient,
) -> None:
    """product_not_found MCP errors raise KaprukaNotFoundError."""
    mcp_client.call_tool = AsyncMock(  # type: ignore[method-assign]
        return_value="Error (product_not_found): Product not found"
    )

    with pytest.raises(KaprukaNotFoundError) as exc_info:
        await get_product(mcp_client, product_id="invalid-id")

    assert exc_info.value.code == "product_not_found"


async def test_get_product_validates_product_id_before_mcp_call(
    mcp_client: MCPHttpClient,
) -> None:
    """GetProductInput rejects product_id shorter than 3 characters."""
    with pytest.raises(ValueError, match="at least 3"):
        await get_product(mcp_client, product_id="ab")

    mcp_client.call_tool.assert_not_awaited()  # type: ignore[attr-defined]
