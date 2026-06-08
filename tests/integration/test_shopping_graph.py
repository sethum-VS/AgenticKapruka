"""End-to-end integration tests for the compiled shopping StateGraph."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.analyze_intent import IntentClassification
from graphs.nodes.generate_response import AssistantReply
from graphs.shopping_graph import ShoppingGraphDeps, build_shopping_graph, initial_shopping_state
from graphs.state import AgentState
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import SearchProductsOutput, TrackOrderOutput

_CLIENT_IP = "203.0.113.42"
_SESSION_ID = "sess-shopping-graph-001"

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[],
    next_cursor=None,
    applied_filters={"q": "birthday cake for mom", "limit": 10, "in_stock_only": False},
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

_SEARCH_TOOL_RESULTS = {
    SEARCH_PRODUCTS_TOOL: {
        "results": [
            {
                "id": "cake00ka002034",
                "name": "Chocolate Birthday Cake",
                "summary": "Rich chocolate layers.",
                "price": {"amount": 4500.0, "currency": "LKR"},
                "compare_at_price": None,
                "in_stock": True,
                "stock_level": "high",
                "image_url": "https://example.com/cake.jpg",
                "category": {
                    "id": "cat_cakes",
                    "name": "Birthday",
                    "slug": "birthday",
                },
                "rating": None,
                "ships_internationally": False,
                "url": "https://www.kapruka.com/cake",
            },
        ],
        "next_cursor": None,
        "applied_filters": {
            "q": "birthday cake for mom",
            "limit": 10,
            "in_stock_only": False,
        },
    },
}


def _mock_genai_client() -> MagicMock:
    """Gemini client returning discovery intent then assistant reply."""
    mock_client = MagicMock()

    intent_response = MagicMock()
    intent_response.parsed = IntentClassification(intent="discovery")
    intent_response.text = '{"intent": "discovery"}'

    reply_response = MagicMock()
    reply_response.parsed = AssistantReply(
        message="I found Chocolate Birthday Cake (LKR 4,500) for your mom's birthday.",
    )
    reply_response.text = reply_response.parsed.model_dump_json()

    mock_client.models.generate_content.side_effect = [intent_response, reply_response]
    return mock_client


def _mock_kapruka_service() -> AsyncMock:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    return mock_service


@pytest.fixture
def graph_deps() -> ShoppingGraphDeps:
    return ShoppingGraphDeps(
        kapruka_service=_mock_kapruka_service(),
        client_ip=_CLIENT_IP,
        genai_client=_mock_genai_client(),
    )


@pytest.mark.asyncio
async def test_shopping_graph_end_to_end_discovery_flow(graph_deps: ShoppingGraphDeps) -> None:
    """Graph runs analyze → hybrid stub → MCP search → generate_response with mocks."""
    graph = build_shopping_graph(deps=graph_deps)
    state: AgentState = initial_shopping_state(
        message="birthday cake for mom",
        session_id=_SESSION_ID,
    )

    result = await graph.ainvoke(state)

    assert result["intent"] == "discovery"
    assert result["hybrid_context"] == {}
    assert result["tool_call_count"] == 1
    assert SEARCH_PRODUCTS_TOOL in (result.get("tool_results") or {})
    assert "Chocolate Birthday Cake" in (result.get("response_html") or "")

    kapruka_service = graph_deps.kapruka_service
    assert isinstance(kapruka_service, AsyncMock)
    kapruka_service.search_products.assert_awaited_once_with(
        _CLIENT_IP,
        q="birthday cake for mom",
        currency="LKR",
    )

    genai_client = graph_deps.genai_client
    assert isinstance(genai_client, MagicMock)
    assert genai_client.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_shopping_graph_tracking_skips_hybrid_context(graph_deps: ShoppingGraphDeps) -> None:
    """Tracking intent routes directly to call_mcp_tools without hybrid retrieval."""
    kapruka_service = graph_deps.kapruka_service
    assert isinstance(kapruka_service, AsyncMock)
    kapruka_service.track_order.return_value = _TRACK_OUTPUT

    genai_client = graph_deps.genai_client
    assert isinstance(genai_client, MagicMock)
    tracking_response = MagicMock()
    tracking_response.parsed = IntentClassification(intent="tracking")
    tracking_response.text = '{"intent": "tracking"}'
    genai_client.models.generate_content.side_effect = [tracking_response]

    graph = build_shopping_graph(deps=graph_deps)
    result = await graph.ainvoke(
        initial_shopping_state(
            message="where is order VIMP34456CB2",
            session_id=_SESSION_ID,
        ),
    )

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
    assert genai_client.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_shopping_graph_node_order_via_stream_events(graph_deps: ShoppingGraphDeps) -> None:
    """astream_events confirms rigid node order for discovery intent."""
    graph = build_shopping_graph(deps=graph_deps)
    config: dict[str, Any] = {"configurable": {"thread_id": "thread-order-test"}}
    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "session_id": _SESSION_ID,
    }

    node_names: list[str] = []
    async for event in graph.astream_events(state, config, version="v2"):
        if event.get("event") == "on_chain_start" and event.get("name") in {
            "load_zep_memory",
            "analyze_intent",
            "retrieve_hybrid_context",
            "call_mcp_tools",
            "generate_response",
            "zep_memory_write",
        }:
            node_names.append(str(event["name"]))

    assert node_names == [
        "load_zep_memory",
        "analyze_intent",
        "retrieve_hybrid_context",
        "call_mcp_tools",
        "generate_response",
        "zep_memory_write",
    ]
