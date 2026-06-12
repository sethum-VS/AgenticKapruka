"""End-to-end integration tests for the compiled shopping StateGraph."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from graphs.nodes.agent_loop import AgentPlannerStep
from graphs.nodes.analyze_intent import IntentClassification
from graphs.nodes.generate_response import AssistantReply
from graphs.shopping_graph import ShoppingGraphDeps, build_shopping_graph, initial_shopping_state
from graphs.state import AgentState
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import (
    GetProductOutput,
    Money,
    ProductAttributes,
    ProductShipping,
    SearchProductsOutput,
    TrackOrderOutput,
)

_CLIENT_IP = "203.0.113.42"
_SESSION_ID = "sess-shopping-graph-001"

_GRAPH_NODE_NAMES = frozenset(
    {
        "load_zep_memory",
        "analyze_intent",
        "retrieve_hybrid_context",
        "agent_loop",
        "call_mcp_tools",
        "generate_response",
        "zep_memory_write",
    },
)

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[],
    next_cursor=None,
    applied_filters={"q": "cakes", "limit": 10, "in_stock_only": False},
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

_TRACK_OUTPUT = TrackOrderOutput.model_validate(
    {
        "order_number": "VIMP34456CB2",
        "pnref": "12345678901",
        "status": "shipped",
        "status_display": "Out for Delivery",
        "order_date": "June 5, 2026",
        "delivery_date": "June 7, 2026",
        "shipped_date": "June 6, 2026",
        "amount": "15500.00",
        "payment_method": "Visa",
        "comments": None,
        "recipient": {
            "name": "Ada Lovelace",
            "phone": "0771234567",
            "address": "123 Galle Road",
            "city": "Colombo 03",
        },
        "greeting_message": None,
        "special_instructions": None,
        "progress": [{"step": "shipped", "timestamp": "June 6, 2026 08:00 AM"}],
        "live_tracking_available": False,
        "has_delivery_video": False,
        "has_delivery_photo": False,
        "items": [],
    },
)


def _mock_genai_client(*, intent: str = "discovery") -> MagicMock:
    """Gemini client returning intent classification and assistant reply on demand."""
    mock_client = MagicMock()

    def generate_content(
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
        **kwargs: Any,
    ) -> MagicMock:
        _ = model, contents, kwargs
        response = MagicMock()
        if config is not None and config.response_schema is IntentClassification:
            response.parsed = IntentClassification(intent=intent)  # type: ignore[arg-type]
            response.text = json.dumps({"intent": intent})
            return response
        if config is not None and config.response_schema is AssistantReply:
            message = "Happy to help with your Kapruka gift search."
            response.parsed = AssistantReply(message=message)
            response.text = json.dumps({"message": message})
            return response
        response.parsed = IntentClassification(intent=intent)  # type: ignore[arg-type]
        response.text = json.dumps({"intent": intent})
        return response

    mock_client.models.generate_content.side_effect = generate_content
    return mock_client


def _mock_kapruka_service() -> AsyncMock:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    mock_service.get_product.return_value = _GET_PRODUCT_OUTPUT
    return mock_service


async def _collect_graph_node_names(
    graph: Any,
    state: AgentState,
    *,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Return ordered LangGraph node names from astream_events."""
    node_names: list[str] = []
    resolved_config = config or {"configurable": {"thread_id": "thread-order-test"}}
    async for event in graph.astream_events(state, resolved_config, version="v2"):
        if event.get("event") == "on_chain_start" and event.get("name") in _GRAPH_NODE_NAMES:
            node_names.append(str(event["name"]))
    return node_names


@pytest.fixture
def graph_deps() -> ShoppingGraphDeps:
    return ShoppingGraphDeps(
        kapruka_service=_mock_kapruka_service(),
        client_ip=_CLIENT_IP,
        genai_client=_mock_genai_client(),
    )


@pytest.mark.asyncio
async def test_shopping_graph_cakes_search_via_agent_loop(graph_deps: ShoppingGraphDeps) -> None:
    """Discovery cakes routes hybrid context → agent_loop search → generate_response."""
    planner_calls = 0

    def _planner_side_effect(*_args: object, **_kwargs: object) -> AgentPlannerStep:
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return AgentPlannerStep(
                action="call_tool",
                tool_name=SEARCH_PRODUCTS_TOOL,
                tool_args={"q": "cakes"},
                rationale="search cakes",
            )
        return AgentPlannerStep(action="finish", rationale="products found")

    graph = build_shopping_graph(deps=graph_deps)
    state: AgentState = initial_shopping_state(
        message="cakes for birthday",
        session_id=_SESSION_ID,
    )

    with patch(
        "graphs.nodes.agent_loop._plan_next_step_sync",
        side_effect=_planner_side_effect,
    ):
        result = await graph.ainvoke(state)
        node_names = await _collect_graph_node_names(graph, state)

    assert result["intent"] == "discovery"
    tool_trace = result.get("tool_trace") or []
    assert len(tool_trace) == 1
    assert tool_trace[0]["name"] == SEARCH_PRODUCTS_TOOL
    assert result.get("tool_call_count") == 1
    assert result.get("agent_loop_done") is True

    kapruka_service = graph_deps.kapruka_service
    assert isinstance(kapruka_service, AsyncMock)
    kapruka_service.search_products.assert_awaited_once_with(
        _CLIENT_IP,
        q="cakes",
        currency="LKR",
    )

    assert node_names == [
        "load_zep_memory",
        "analyze_intent",
        "retrieve_hybrid_context",
        "agent_loop",
        "generate_response",
        "zep_memory_write",
    ]
    assert "call_mcp_tools" not in node_names


@pytest.mark.asyncio
async def test_shopping_graph_thanks_skips_tools_via_agent_loop(
    graph_deps: ShoppingGraphDeps,
) -> None:
    """General thanks routes through agent_loop with planner finish and no Kapruka calls."""
    general_deps = ShoppingGraphDeps(
        kapruka_service=graph_deps.kapruka_service,
        client_ip=graph_deps.client_ip,
        genai_client=_mock_genai_client(intent="general"),
    )
    graph = build_shopping_graph(deps=general_deps)
    state: AgentState = initial_shopping_state(
        message="thanks!",
        session_id=_SESSION_ID,
    )

    with patch(
        "graphs.nodes.agent_loop._plan_next_step_sync",
        return_value=AgentPlannerStep(
            action="finish",
            refined_intent="general",
            rationale="no tools needed",
        ),
    ):
        result = await graph.ainvoke(state)
        node_names = await _collect_graph_node_names(graph, state)

    assert result["intent"] == "general"
    assert result.get("tool_trace") == []
    assert result.get("tool_call_count") in (None, 0)
    assert result.get("agent_loop_done") is True

    kapruka_service = graph_deps.kapruka_service
    assert isinstance(kapruka_service, AsyncMock)
    kapruka_service.search_products.assert_not_awaited()
    kapruka_service.get_product.assert_not_awaited()
    kapruka_service.list_categories.assert_not_awaited()
    kapruka_service.track_order.assert_not_awaited()

    assert "agent_loop" in node_names
    assert "call_mcp_tools" not in node_names


@pytest.mark.asyncio
async def test_shopping_graph_tracking_skips_hybrid_context_and_agent_loop(
    graph_deps: ShoppingGraphDeps,
) -> None:
    """Tracking intent routes directly to call_mcp_tools without hybrid retrieval or agent loop."""
    kapruka_service = graph_deps.kapruka_service
    assert isinstance(kapruka_service, AsyncMock)
    kapruka_service.track_order.return_value = _TRACK_OUTPUT

    graph_deps = ShoppingGraphDeps(
        kapruka_service=kapruka_service,
        client_ip=graph_deps.client_ip,
        genai_client=_mock_genai_client(intent="tracking"),
    )
    graph = build_shopping_graph(deps=graph_deps)
    state: AgentState = initial_shopping_state(
        message="where is order VIMP34456CB2",
        session_id=_SESSION_ID,
    )
    result = await graph.ainvoke(state)

    assert result["intent"] == "tracking"
    assert result.get("hybrid_context") is None
    assert result.get("tool_call_count") == 1
    tool_results = result.get("tool_results") or {}
    assert TRACK_ORDER_TOOL in tool_results
    assert SEARCH_PRODUCTS_TOOL not in tool_results
    assert "VIMP34456CB2" in (result.get("response_html") or "")
    kapruka_service.search_products.assert_not_awaited()
    kapruka_service.track_order.assert_awaited_once_with(
        _CLIENT_IP,
        order_number="VIMP34456CB2",
    )
    genai_client = graph_deps.genai_client
    assert isinstance(genai_client, MagicMock)
    # Phase 2: tracking is guard + template only — no intent LLM or planner.
    assert genai_client.models.generate_content.call_count == 0

    node_names = await _collect_graph_node_names(graph, state)
    assert node_names == [
        "load_zep_memory",
        "analyze_intent",
        "call_mcp_tools",
        "generate_response",
        "zep_memory_write",
    ]
    assert "agent_loop" not in node_names
    assert "retrieve_hybrid_context" not in node_names


@pytest.mark.asyncio
async def test_shopping_graph_product_id_fast_path_skips_agent_loop(
    graph_deps: ShoppingGraphDeps,
) -> None:
    """Product ID in message routes to call_mcp_tools get_product without agent_loop."""
    graph = build_shopping_graph(deps=graph_deps)
    state: AgentState = initial_shopping_state(
        message="tell me about cake00ka002034",
        session_id=_SESSION_ID,
    )

    result = await graph.ainvoke(state)

    assert result["intent"] == "discovery"
    tool_results = result.get("tool_results") or {}
    assert GET_PRODUCT_TOOL in tool_results
    assert SEARCH_PRODUCTS_TOOL not in tool_results
    assert result.get("tool_call_count") == 1
    assert result.get("tool_trace") in (None, [])

    kapruka_service = graph_deps.kapruka_service
    assert isinstance(kapruka_service, AsyncMock)
    kapruka_service.get_product.assert_awaited_once_with(
        _CLIENT_IP,
        product_id="cake00ka002034",
        currency="LKR",
    )
    kapruka_service.search_products.assert_not_awaited()

    node_names = await _collect_graph_node_names(graph, state)
    assert node_names == [
        "load_zep_memory",
        "analyze_intent",
        "call_mcp_tools",
        "generate_response",
        "zep_memory_write",
    ]
    assert "agent_loop" not in node_names
    assert "retrieve_hybrid_context" not in node_names
