"""Unit tests for graphs.nodes.call_mcp_tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.call_mcp_tools import call_mcp_tools, select_tool_calls
from graphs.state import AgentState
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import (
    GetProductOutput,
    ListCategoriesOutput,
    Money,
    ProductAttributes,
    ProductShipping,
    SearchProductsOutput,
    TrackOrderOutput,
)
from lib.neo4j.hybrid_context import RewrittenSearchQuery

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
async def test_call_mcp_tools_injects_session_currency_into_explicit_tool_calls() -> None:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT

    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "intent": "discovery",
        "currency": "USD",
        "tool_calls": [
            {
                "name": SEARCH_PRODUCTS_TOOL,
                "args": {"q": "birthday cake for mom"},
            },
        ],
        "session_id": "sess-mcp-currency-001",
    }

    await call_mcp_tools(state, kapruka_service=mock_service, client_ip=_CLIENT_IP)

    mock_service.search_products.assert_awaited_once_with(
        _CLIENT_IP,
        q="birthday cake for mom",
        currency="USD",
    )


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


def test_select_tool_calls_tracking_returns_empty_without_order_number() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="where is my order?")],
        "intent": "tracking",
    }
    assert select_tool_calls(state) == []


def test_select_tool_calls_tracking_extracts_order_number() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="where is order VIMP34456CB2")],
        "intent": "tracking",
    }
    selected = select_tool_calls(state)
    assert len(selected) == 1
    assert selected[0]["name"] == TRACK_ORDER_TOOL
    assert selected[0]["args"]["order_number"] == "VIMP34456CB2"


def test_select_tool_calls_applies_category_preference_from_hybrid_context() -> None:
    """Discovery search should include category hint from Zep preferences."""
    state: AgentState = {
        "messages": [HumanMessage(content="something nice for her")],
        "intent": "discovery",
        "hybrid_context": {
            "preferences": {"favorite_category": "Birthday"},
            "hints": {"category": "Birthday"},
        },
    }

    selected = select_tool_calls(state)

    assert len(selected) == 1
    assert selected[0]["name"] == SEARCH_PRODUCTS_TOOL
    assert selected[0]["args"]["q"] == "something nice for her"
    assert selected[0]["args"]["category"] == "Birthday"


@pytest.mark.asyncio
async def test_call_mcp_tools_applies_category_preference_in_search() -> None:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT

    state: AgentState = {
        "messages": [HumanMessage(content="something nice for her")],
        "intent": "discovery",
        "session_id": "sess-mcp-pref-001",
        "hybrid_context": {
            "preferences": {"favorite_category": "Birthday", "currency": "USD"},
            "hints": {"category": "Birthday", "currency": "USD"},
        },
    }

    await call_mcp_tools(state, kapruka_service=mock_service, client_ip=_CLIENT_IP)

    mock_service.search_products.assert_awaited_once_with(
        _CLIENT_IP,
        q="something nice for her",
        currency="USD",
        category="Birthday",
    )


@pytest.mark.asyncio
async def test_call_mcp_tools_tracking_replaces_stale_discovery_results() -> None:
    """Tracking with an order number invokes track_order and drops stale search payloads."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.track_order.return_value = TrackOrderOutput.model_validate(
        {
            "order_number": "VIMP34456CB2",
            "pnref": "1",
            "status": "confirmed",
            "status_display": "Confirmed",
            "order_date": "June 5, 2026",
            "delivery_date": "June 7, 2026",
            "amount": "1000.00",
            "payment_method": "Visa",
            "recipient": {
                "name": "Test",
                "phone": "0771234567",
                "address": "1 Road",
                "city": "Colombo 03",
            },
            "progress": [],
            "live_tracking_available": False,
            "has_delivery_video": False,
            "has_delivery_photo": False,
            "items": [],
        }
    )
    state: AgentState = {
        "messages": [HumanMessage(content="where is order VIMP34456CB2")],
        "intent": "tracking",
        "tool_results": {
            SEARCH_PRODUCTS_TOOL: {
                "results": [{"id": "stale-cake", "name": "Stale Birthday Cake"}],
            },
        },
    }

    result = await call_mcp_tools(
        state,
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
    )

    mock_service.track_order.assert_awaited_once_with(
        _CLIENT_IP,
        order_number="VIMP34456CB2",
    )
    assert set(result["tool_results"].keys()) == {TRACK_ORDER_TOOL}
    assert SEARCH_PRODUCTS_TOOL not in result["tool_results"]
    mock_service.search_products.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_mcp_tools_tracking_without_number_clears_stale_tool_results() -> None:
    """Tracking without an order number must not carry prior-turn MCP payloads into state."""
    mock_service = AsyncMock(spec=KaprukaService)
    state: AgentState = {
        "messages": [HumanMessage(content="where is my order?")],
        "intent": "tracking",
        "tool_results": {
            SEARCH_PRODUCTS_TOOL: {
                "results": [{"id": "stale-cake", "name": "Stale Birthday Cake"}],
            },
        },
    }

    result = await call_mcp_tools(
        state,
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
    )

    assert result == {"tool_results": {}}
    mock_service.track_order.assert_not_awaited()


def test_select_tool_calls_graph_birthday_context_sets_category_filter() -> None:
    """Graph-informed hybrid_context should map top category to MCP search filter."""
    state: AgentState = {
        "messages": [HumanMessage(content="cake for mom")],
        "intent": "discovery",
        "hybrid_context": {
            "hints": {"category": "Birthday", "occasion": "Birthday"},
            "vector_hits": [
                {
                    "id": "category:cakes",
                    "score": 0.91,
                    "display_name": "Cakes",
                },
            ],
            "occasions": [{"display_name": "Birthday", "hop": 1}],
        },
    }

    selected = select_tool_calls(state)

    assert len(selected) == 1
    assert selected[0]["name"] == SEARCH_PRODUCTS_TOOL
    assert selected[0]["args"]["category"] == "Birthday"
    assert selected[0]["args"]["q"] == "cake for mom"
    assert " Birthday" not in selected[0]["args"]["q"]


@pytest.mark.asyncio
async def test_call_mcp_tools_graph_birthday_context_invokes_search_with_category() -> None:
    """Discovery search uses category filter and Gemini rewrite instead of raw concatenation."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    mock_genai = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = RewrittenSearchQuery(q="birthday cake for mom")
    mock_response.text = '{"q": "birthday cake for mom"}'
    mock_genai.models.generate_content.return_value = mock_response

    state: AgentState = {
        "messages": [HumanMessage(content="cake for mom")],
        "intent": "discovery",
        "session_id": "sess-mcp-graph-049",
        "hybrid_context": {
            "hints": {"category": "Birthday", "occasion": "Birthday"},
            "vector_hits": [
                {
                    "id": "category:cakes",
                    "score": 0.91,
                    "display_name": "Cakes",
                },
            ],
        },
    }

    await call_mcp_tools(
        state,
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=mock_genai,
    )

    mock_service.search_products.assert_awaited_once_with(
        _CLIENT_IP,
        q="birthday cake for mom",
        currency="LKR",
        category="Birthday",
    )
    mock_genai.models.generate_content.assert_called_once()


def test_select_tool_calls_skips_occasion_augment_when_confidence_low() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="something elegant")],
        "intent": "discovery",
        "hybrid_context": {
            "hints": {"category": "Flowers", "occasion": "Wedding"},
            "vector_hits": [{"id": "category:flowers", "score": 0.4, "display_name": "Flowers"}],
        },
    }

    selected = select_tool_calls(state)

    assert selected[0]["args"]["category"] == "Flowers"
    assert selected[0]["args"]["q"] == "something elegant"


@pytest.mark.asyncio
async def test_call_mcp_tools_does_not_merge_prior_turn_tool_results() -> None:
    """Each turn stores only the current invocation's MCP outputs."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.list_categories.return_value = _LIST_CATEGORIES_OUTPUT

    state: AgentState = {
        "messages": [HumanMessage(content="what categories do you have")],
        "intent": "general",
        "tool_results": {
            SEARCH_PRODUCTS_TOOL: {
                "results": [{"id": "stale-cake", "name": "Stale Birthday Cake"}],
            },
        },
    }

    result = await call_mcp_tools(
        state,
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
    )

    assert set(result["tool_results"].keys()) == {LIST_CATEGORIES_TOOL}
    assert SEARCH_PRODUCTS_TOOL not in result["tool_results"]
