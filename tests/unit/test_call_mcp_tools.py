"""Unit tests for graphs.nodes.call_mcp_tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.call_mcp_tools import call_mcp_tools, select_tool_calls
from graphs.state import AgentState
from lib.kapruka.errors import KaprukaNotFoundError
from lib.kapruka.service import KaprukaService
from lib.kapruka.tool_executor import canonical_tool_args_for_dedup, invoke_tool
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
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
async def test_call_mcp_tools_discovery_without_product_id_selects_no_tools() -> None:
    """Discovery catalog turns route through agent_loop — call_mcp_tools stays idle."""
    mock_service = AsyncMock(spec=KaprukaService)

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

    mock_service.search_products.assert_not_awaited()
    assert result == {"tool_results": {}}


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
async def test_call_mcp_tools_discovery_does_not_increment_tool_call_count() -> None:
    """Idle discovery path must not bump tool_call_count."""
    mock_service = AsyncMock(spec=KaprukaService)

    state: AgentState = {
        "messages": [HumanMessage(content="roses bouquet")],
        "intent": "discovery",
        "tool_call_count": 2,
        "session_id": "sess-mcp-004",
    }

    result = await call_mcp_tools(state, kapruka_service=mock_service, client_ip=_CLIENT_IP)

    assert result == {"tool_results": {}}
    assert "tool_call_count" not in result


def test_select_tool_calls_discovery_with_product_id_prefers_get_product() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="details for cake00ka002034 please")],
        "intent": "discovery",
    }

    selected = select_tool_calls(state)

    assert len(selected) == 1
    assert selected[0]["name"] == GET_PRODUCT_TOOL
    assert selected[0]["args"]["product_id"] == "cake00ka002034"


def test_select_tool_calls_general_with_product_id_prefers_get_product() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="cake00ka002034")],
        "intent": "general",
    }

    selected = select_tool_calls(state)

    assert len(selected) == 1
    assert selected[0]["name"] == GET_PRODUCT_TOOL
    assert selected[0]["args"]["product_id"] == "cake00ka002034"


@pytest.mark.asyncio
async def test_call_mcp_tools_general_product_id_invokes_get_product_without_search() -> None:
    """Product ID fast-path issues kapruka_get_product only — no hybrid context or search."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.get_product.return_value = _GET_PRODUCT_OUTPUT

    state: AgentState = {
        "messages": [HumanMessage(content="cake00ka002034")],
        "intent": "general",
        "session_id": "sess-mcp-product-id-fast",
    }

    result = await call_mcp_tools(state, kapruka_service=mock_service, client_ip=_CLIENT_IP)

    mock_service.get_product.assert_awaited_once_with(
        _CLIENT_IP,
        product_id="cake00ka002034",
        currency="LKR",
    )
    mock_service.search_products.assert_not_called()
    mock_service.list_categories.assert_not_called()
    assert result["tool_call_count"] == 1
    assert result["tool_results"][GET_PRODUCT_TOOL]["id"] == "cake00ka002034"


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


def test_select_tool_calls_discovery_ignores_hybrid_context_category_hints() -> None:
    """Hybrid context hints are planner-only on discovery — no MCP arg injection."""
    state: AgentState = {
        "messages": [HumanMessage(content="something nice for her")],
        "intent": "discovery",
        "hybrid_context": {
            "preferences": {"favorite_category": "Birthday"},
            "hints": {"category": "Birthday"},
        },
    }

    assert select_tool_calls(state) == []


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


def test_select_tool_calls_discovery_graph_context_does_not_inject_search() -> None:
    """PRD-049 graph-informed filter injection no longer applies on discovery loop route."""
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

    assert select_tool_calls(state) == []


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


def test_select_tool_calls_discovery_kandy_delivery_defers_to_agent_loop() -> None:
    """Delivery validation on discovery turns is planner-driven in agent_loop."""
    state: AgentState = {
        "messages": [HumanMessage(content="Machan, can you deliver to Kandy on Sunday?")],
        "intent": "discovery",
        "intent_metadata": {
            "is_situational": False,
            "detected_vernacular": "tanglish",
            "requires_delivery_validation": True,
            "target_city": "Kandy",
            "budget_max": None,
        },
    }

    assert select_tool_calls(state) == []


def test_select_tool_calls_discovery_product_id_with_delivery_still_binds_check() -> None:
    """Product-ID fast-path retains proactive delivery check when metadata requires it."""
    state: AgentState = {
        "messages": [HumanMessage(content="cake00ka002034 deliver to Kandy")],
        "intent": "discovery",
        "intent_metadata": {
            "is_situational": False,
            "detected_vernacular": "en",
            "requires_delivery_validation": True,
            "target_city": "Kandy",
            "budget_max": None,
        },
    }

    selected = select_tool_calls(state)

    assert len(selected) == 2
    assert selected[0]["name"] == GET_PRODUCT_TOOL
    assert selected[0]["args"]["product_id"] == "cake00ka002034"
    assert selected[1]["name"] == CHECK_DELIVERY_TOOL
    assert selected[1]["args"] == {"city": "Kandy"}


@pytest.mark.asyncio
async def test_invoke_tool_tracking_serializes_order_status() -> None:
    """Shared executor validates tracking args and returns serialized MCP output."""
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

    result = await invoke_tool(
        TRACK_ORDER_TOOL,
        {"order_number": "VIMP34456CB2"},
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
    )

    mock_service.track_order.assert_awaited_once_with(
        _CLIENT_IP,
        order_number="VIMP34456CB2",
    )
    assert result["order_number"] == "VIMP34456CB2"
    assert result["status"] == "confirmed"


@pytest.mark.asyncio
async def test_invoke_tool_rejects_invalid_tracking_args() -> None:
    """Pydantic validation failures map to structured executor errors."""
    mock_service = AsyncMock(spec=KaprukaService)

    result = await invoke_tool(
        TRACK_ORDER_TOOL,
        {"order_number": "ab"},
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
    )

    assert result["error"] == "validation_error"
    mock_service.track_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_invoke_tool_maps_kapruka_errors() -> None:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.track_order.side_effect = KaprukaNotFoundError(
        "order_not_found",
        "Order not found",
    )

    result = await invoke_tool(
        TRACK_ORDER_TOOL,
        {"order_number": "VIMP34456CB2"},
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
    )

    assert result["error"] == "order_not_found"
    assert "message" in result


@pytest.mark.asyncio
async def test_call_mcp_tools_delegates_invocation_to_shared_executor() -> None:
    """Tracking path routes through lib.kapruka.tool_executor.invoke_tool."""
    mock_service = AsyncMock(spec=KaprukaService)
    state: AgentState = {
        "messages": [HumanMessage(content="where is order VIMP34456CB2")],
        "intent": "tracking",
    }

    with patch(
        "graphs.nodes.call_mcp_tools.invoke_tool",
        new_callable=AsyncMock,
        return_value={"order_number": "VIMP34456CB2", "status": "confirmed"},
    ) as mock_invoke:
        result = await call_mcp_tools(
            state,
            kapruka_service=mock_service,
            client_ip=_CLIENT_IP,
        )

    mock_invoke.assert_awaited_once()
    call_kwargs = mock_invoke.await_args.kwargs
    assert call_kwargs["kapruka_service"] is mock_service
    assert call_kwargs["client_ip"] == _CLIENT_IP
    assert result["tool_results"][TRACK_ORDER_TOOL]["status"] == "confirmed"


@pytest.mark.asyncio
async def test_invoke_tool_normalizes_search_query_alias() -> None:
    """Planner tool_args using query are coerced to q before validation."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT

    result = await invoke_tool(
        SEARCH_PRODUCTS_TOOL,
        {"query": "cakes"},
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        currency="LKR",
    )

    assert "error" not in result
    mock_service.search_products.assert_awaited_once()
    assert mock_service.search_products.await_args.kwargs["q"] == "cakes"


@pytest.mark.asyncio
async def test_invoke_tool_normalizes_search_category_id_alias() -> None:
    """Planner tool_args using category_id are coerced to category before validation."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT

    result = await invoke_tool(
        SEARCH_PRODUCTS_TOOL,
        {"q": "cakes", "category_id": "Birthday"},
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        currency="LKR",
    )

    assert "error" not in result
    assert mock_service.search_products.await_args.kwargs["category"] == "Birthday"


def test_canonical_tool_args_for_dedup_ignores_currency() -> None:
    """Duplicate detection treats currency injection as the same search invocation."""
    left = canonical_tool_args_for_dedup(SEARCH_PRODUCTS_TOOL, {"q": "cakes", "currency": "LKR"})
    right = canonical_tool_args_for_dedup(SEARCH_PRODUCTS_TOOL, {"q": "cakes"})
    assert left == right


def test_canonical_tool_args_for_dedup_normalizes_query_alias() -> None:
    """Planner query alias must dedupe identically to canonical q."""
    left = canonical_tool_args_for_dedup(SEARCH_PRODUCTS_TOOL, {"q": "cakes"})
    right = canonical_tool_args_for_dedup(SEARCH_PRODUCTS_TOOL, {"query": "cakes"})
    assert left == right
