"""Unit tests for graphs.nodes.call_mcp_tools."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.call_mcp_tools import call_mcp_tools, select_tool_calls
from graphs.state import AgentState
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.types import (
    GetProductOutput,
    ListCategoriesOutput,
    Money,
    ProductAttributes,
    ProductShipping,
    SearchProductsOutput,
)

_CLIENT_IP = "203.0.113.42"

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[],
    next_cursor=None,
    applied_filters={"q": "birthday cake for mom", "limit": 10, "in_stock_only": False},
)

_GET_PRODUCT_OUTPUT = GetProductOutput(
    id="cake00ka002034",
    name="Chocolate Birthday Cake",
    description="Rich chocolate cake.",
    summary="Rich chocolate cake.",
    price=Money(amount=4500.0, currency="LKR"),
    compare_at_price=None,
    in_stock=True,
    stock_level="high",
    category={"id": "cat_cakes", "name": "Birthday", "slug": "birthday", "path": None},
    variants=[],
    images=[],
    attributes=ProductAttributes(),
    shipping=ProductShipping(
        ships_from="Sri Lanka",
        ships_internationally=False,
        restricted_countries=[],
    ),
    rating=None,
    url="https://www.kapruka.com/cake",
)

_LIST_CATEGORIES_OUTPUT = ListCategoriesOutput(categories=[])


@pytest.mark.asyncio
async def test_call_mcp_tools_discovery_invokes_search_products() -> None:
    """Discovery intent without explicit tool_calls triggers kapruka_search_products."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT

    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "intent": "discovery",
        "session_id": "sess-mcp-001",
    }

    result = await call_mcp_tools(
        state,
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
    )

    mock_service.search_products.assert_awaited_once_with(
        _CLIENT_IP,
        q="birthday cake for mom",
        currency="LKR",
    )
    assert result["tool_call_count"] == 1
    assert SEARCH_PRODUCTS_TOOL in result["tool_results"]
    assert result["tool_results"][SEARCH_PRODUCTS_TOOL]["results"] == []


@pytest.mark.asyncio
async def test_call_mcp_tools_general_invokes_list_categories() -> None:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.list_categories.return_value = _LIST_CATEGORIES_OUTPUT

    state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "intent": "general",
        "session_id": "sess-mcp-002",
    }

    result = await call_mcp_tools(state, kapruka_service=mock_service, client_ip=_CLIENT_IP)

    mock_service.list_categories.assert_awaited_once_with(_CLIENT_IP, depth=1)
    assert result["tool_call_count"] == 1
    assert LIST_CATEGORIES_TOOL in result["tool_results"]


@pytest.mark.asyncio
async def test_call_mcp_tools_explicit_tool_calls_override_intent() -> None:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.get_product.return_value = _GET_PRODUCT_OUTPUT

    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "intent": "discovery",
        "tool_calls": [
            {
                "name": GET_PRODUCT_TOOL,
                "args": {"product_id": "cake00ka002034", "currency": "USD"},
            },
        ],
        "session_id": "sess-mcp-003",
    }

    result = await call_mcp_tools(state, kapruka_service=mock_service, client_ip=_CLIENT_IP)

    mock_service.get_product.assert_awaited_once_with(
        _CLIENT_IP,
        product_id="cake00ka002034",
        currency="USD",
    )
    mock_service.search_products.assert_not_called()
    assert result["tool_call_count"] == 1
    assert result["tool_results"][GET_PRODUCT_TOOL]["id"] == "cake00ka002034"


@pytest.mark.asyncio
async def test_call_mcp_tools_increments_existing_tool_call_count() -> None:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT

    state: AgentState = {
        "messages": [HumanMessage(content="roses bouquet")],
        "intent": "discovery",
        "tool_call_count": 2,
        "session_id": "sess-mcp-004",
    }

    result = await call_mcp_tools(state, kapruka_service=mock_service, client_ip=_CLIENT_IP)

    assert result["tool_call_count"] == 3


def test_select_tool_calls_discovery_with_product_id_prefers_get_product() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="details for cake00ka002034 please")],
        "intent": "discovery",
    }

    selected = select_tool_calls(state)

    assert len(selected) == 1
    assert selected[0]["name"] == GET_PRODUCT_TOOL
    assert selected[0]["args"]["product_id"] == "cake00ka002034"


def test_select_tool_calls_tracking_returns_empty_without_explicit_calls() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="where is my order?")],
        "intent": "tracking",
    }
    assert select_tool_calls(state) == []
