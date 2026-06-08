"""Unit tests for deterministic Kapruka MCP mock fixture."""

from __future__ import annotations

import json

import pytest
from tests.fixtures.mcp_mock import (
    ALL_MOCK_TOOL_NAMES,
    MockMCPHttpClient,
    mock_mcp_response,
)

from lib.kapruka.tools.create_order import TOOL_NAME as CREATE_ORDER_TOOL
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL


@pytest.mark.parametrize("tool_name", sorted(ALL_MOCK_TOOL_NAMES))
def test_mock_mcp_response_covers_all_seven_tools(tool_name: str) -> None:
    payload = mock_mcp_response(tool_name, {})
    assert isinstance(payload, dict)
    assert payload


async def test_mock_mcp_client_returns_json_for_each_tool() -> None:
    client = await MockMCPHttpClient.connect()
    for tool_name in sorted(ALL_MOCK_TOOL_NAMES):
        raw = await client.call_tool(tool_name, {})
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


async def test_search_products_mock_reflects_query() -> None:
    client = await MockMCPHttpClient.connect()
    raw = await client.call_tool(SEARCH_PRODUCTS_TOOL, {"q": "wedding flowers"})
    payload = json.loads(raw)
    assert payload["applied_filters"]["q"] == "wedding flowers"
    assert payload["results"][0]["name"]


async def test_get_product_mock_uses_product_id() -> None:
    payload = mock_mcp_response(GET_PRODUCT_TOOL, {"product_id": "cake00ka002034"})
    assert payload["id"] == "cake00ka002034"
    assert payload["name"]


async def test_track_order_mock_uses_order_number() -> None:
    payload = mock_mcp_response(TRACK_ORDER_TOOL, {"order_number": "VIMP99887AB1"})
    assert payload["order_number"] == "VIMP99887AB1"
    assert payload["status_display"]


async def test_check_delivery_mock_sets_city_and_date() -> None:
    payload = mock_mcp_response(
        CHECK_DELIVERY_TOOL,
        {"city": "Kandy", "delivery_date": "2026-06-15", "product_id": "cake00ka002034"},
    )
    assert payload["city"] == "Kandy"
    assert payload["checked_date"] == "2026-06-15"
    assert payload["available"] is True
    assert payload["perishable_warning"]


async def test_list_delivery_cities_mock_has_colombo() -> None:
    payload = mock_mcp_response(LIST_CITIES_TOOL, {"query": "Col"})
    names = [city["name"] for city in payload["cities"]]
    assert "Colombo 03" in names


async def test_create_order_mock_has_checkout_url() -> None:
    payload = mock_mcp_response(CREATE_ORDER_TOOL, {})
    assert payload["checkout_url"]
    assert payload["order_ref"]
    assert payload["expires_at"]


async def test_list_categories_mock_has_nested_tree() -> None:
    payload = mock_mcp_response(LIST_CATEGORIES_TOOL, {"depth": 1})
    assert payload["categories"][0]["children"]
