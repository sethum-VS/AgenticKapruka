"""Unit tests for graphs.nodes.agent_loop planner loop and trace summarization."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graphs.model_router import FLASH_MODEL
from graphs.nodes.agent_loop import (
    MAX_ITERATIONS,
    PLANNER_CATEGORY_NODE_LIMIT,
    PLANNER_SEARCH_RESULT_LIMIT,
    AgentPlannerStep,
    agent_loop,
    build_planner_prior_iterations,
    format_planner_prior_iterations,
)
from graphs.state import AgentState, ToolInvocation
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.types import SearchProductsOutput

_CLIENT_IP = "203.0.113.99"

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[],
    next_cursor=None,
    applied_filters={"q": "birthday cake", "limit": 10, "in_stock_only": False},
)


def _make_search_product(index: int) -> dict[str, object]:
    return {
        "id": f"cake{index:03d}",
        "name": f"Cake {index}",
        "summary": f"Summary for cake {index}",
        "description": f"Long description for cake {index}",
        "price": {"amount": 1000.0 + index, "currency": "LKR"},
        "in_stock": index % 2 == 0,
        "stock_level": "high",
        "image_url": f"https://cdn.kapruka.com/cake{index}.jpg",
        "category": {"id": "cat_cakes", "name": "Cakes", "slug": "cakes", "path": None},
        "variants": [{"id": f"v{index}", "name": "Default"}],
        "images": [f"https://cdn.kapruka.com/cake{index}.jpg"],
        "url": f"https://www.kapruka.com/cake{index}",
    }


def test_summarize_search_products_caps_at_five_and_strips_heavy_fields() -> None:
    """30-product search → planner summary has ≤5 items and no image URLs."""
    full_results = [_make_search_product(i) for i in range(30)]
    full_result_payload = {
        "results": full_results,
        "next_cursor": "page-2",
        "applied_filters": {"q": "birthday cake", "limit": 30},
    }
    tool_trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "birthday cake", "limit": 30},
            "result": full_result_payload,
        }
    ]

    assert len(tool_trace[0]["result"]["results"]) == 30

    planner_entries = build_planner_prior_iterations(tool_trace)
    assert len(planner_entries) == 1
    summary = planner_entries[0]["summary"]
    assert isinstance(summary, dict)
    summarized_results = summary["results"]
    assert isinstance(summarized_results, list)
    assert len(summarized_results) <= PLANNER_SEARCH_RESULT_LIMIT

    planner_blob = format_planner_prior_iterations(tool_trace)
    assert "image_url" not in planner_blob
    assert "Summary for cake" not in planner_blob
    assert "Long description for cake" not in planner_blob
    assert "variants" not in planner_blob

    for product in summarized_results:
        assert set(product.keys()) <= {"id", "name", "price", "in_stock"}
        assert "image_url" not in product


def test_full_tool_trace_retains_all_search_products() -> None:
    """Summarization for planner must not mutate stored tool_trace payloads."""
    full_results = [_make_search_product(i) for i in range(30)]
    invocation: ToolInvocation = {
        "name": SEARCH_PRODUCTS_TOOL,
        "args": {"q": "birthday cake"},
        "result": {"results": full_results, "next_cursor": None, "applied_filters": {}},
    }
    tool_trace = [invocation]

    build_planner_prior_iterations(tool_trace)
    format_planner_prior_iterations(tool_trace)

    assert len(tool_trace[0]["result"]["results"]) == 30
    assert tool_trace[0]["result"]["results"][0]["image_url"].startswith("https://")


def test_summarize_get_product_strips_catalog_heavy_fields() -> None:
    tool_trace: list[ToolInvocation] = [
        {
            "name": GET_PRODUCT_TOOL,
            "args": {"product_id": "cake001"},
            "result": {
                "id": "cake001",
                "name": "Chocolate Cake",
                "description": "Rich chocolate layers",
                "summary": "Rich chocolate",
                "price": {"amount": 4500.0, "currency": "LKR"},
                "in_stock": True,
                "images": ["https://cdn.kapruka.com/cake001.jpg"],
                "variants": [{"id": "v1", "name": "2lb"}],
            },
        }
    ]
    summary = build_planner_prior_iterations(tool_trace)[0]["summary"]
    assert summary == {
        "id": "cake001",
        "name": "Chocolate Cake",
        "price": {"amount": 4500.0, "currency": "LKR"},
        "in_stock": True,
    }
    assert "description" not in json.dumps(summary)


def test_summarize_list_categories_caps_nodes_and_includes_ids() -> None:
    categories = [
        {
            "name": "Cakes",
            "url": "https://www.kapruka.com/shop/cakes",
            "children": [
                {"name": "Birthday", "url": "https://www.kapruka.com/shop/cakes/birthday"},
                {"name": "Wedding", "url": "https://www.kapruka.com/shop/cakes/wedding"},
            ],
        },
        {
            "name": "Flowers",
            "url": "https://www.kapruka.com/shop/flowers",
            "children": [
                {"name": "Roses", "url": "https://www.kapruka.com/shop/flowers/roses"},
            ],
        },
    ]
    tool_trace: list[ToolInvocation] = [
        {
            "name": LIST_CATEGORIES_TOOL,
            "args": {"depth": 2},
            "result": {"categories": categories},
        }
    ]
    summary = build_planner_prior_iterations(tool_trace)[0]["summary"]
    assert isinstance(summary, dict)
    nodes = summary["categories"]
    assert isinstance(nodes, list)
    assert len(nodes) <= PLANNER_CATEGORY_NODE_LIMIT
    assert nodes[0] == {"name": "Cakes", "id": "cakes"}
    assert all("name" in node and "id" in node for node in nodes)


def test_summarize_check_delivery_city_and_deliverable_only() -> None:
    tool_trace: list[ToolInvocation] = [
        {
            "name": CHECK_DELIVERY_TOOL,
            "args": {"city": "Kandy"},
            "result": {
                "city": "Kandy",
                "now": "2026-06-11T10:00:00+05:30",
                "checked_date": "2026-06-12",
                "available": True,
                "rate": 450.0,
                "currency": "LKR",
                "reason": None,
            },
        }
    ]
    summary = build_planner_prior_iterations(tool_trace)[0]["summary"]
    assert summary == {"city": "Kandy", "deliverable": True}
    assert "rate" not in json.dumps(summary)


def test_summarize_errors_message_only() -> None:
    tool_trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "x"},
            "result": {"error": "product_not_found", "message": "No products found"},
        }
    ]
    summary = build_planner_prior_iterations(tool_trace)[0]["summary"]
    assert summary == {"error": "product_not_found", "message": "No products found"}


def _base_state() -> AgentState:
    return {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "session_id": "sess-agent-loop-001",
        "currency": "LKR",
    }


def _mock_kapruka_service() -> AsyncMock:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    mock_service.get_product.return_value = _SEARCH_OUTPUT
    mock_service.list_categories.return_value = _SEARCH_OUTPUT
    mock_service.check_delivery.return_value = _SEARCH_OUTPUT
    return mock_service


@pytest.mark.asyncio
async def test_agent_loop_finish_sets_done() -> None:
    """Planner finish action ends the loop immediately with agent_loop_done=True."""
    mock_service = _mock_kapruka_service()
    finish_step = AgentPlannerStep(action="finish", rationale="products found")

    with patch(
        "graphs.nodes.agent_loop._plan_next_step_sync",
        return_value=finish_step,
    ):
        result = await agent_loop(
            _base_state(),
            kapruka_service=mock_service,
            client_ip=_CLIENT_IP,
        )

    assert result["agent_loop_done"] is True
    assert result.get("agent_clarifying_question") is None
    assert result["tool_trace"] == []
    mock_service.search_products.assert_not_called()


@pytest.mark.asyncio
async def test_agent_loop_ask_user_sets_clarifying_question_and_exits() -> None:
    """ask_user stores clarifying question and exits without tool calls."""
    mock_service = _mock_kapruka_service()
    ask_step = AgentPlannerStep(
        action="ask_user",
        rationale="Which city should we deliver to?",
    )

    with patch(
        "graphs.nodes.agent_loop._plan_next_step_sync",
        return_value=ask_step,
    ):
        result = await agent_loop(
            _base_state(),
            kapruka_service=mock_service,
            client_ip=_CLIENT_IP,
        )

    assert result["agent_loop_done"] is True
    assert result["agent_clarifying_question"] == "Which city should we deliver to?"
    assert result["tool_trace"] == []
    mock_service.search_products.assert_not_called()


@pytest.mark.asyncio
async def test_agent_loop_iteration_cap_limits_tool_calls() -> None:
    """Bounded loop runs at most MAX_ITERATIONS planner steps with tool invocations."""
    mock_service = _mock_kapruka_service()
    planner_steps = [
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": f"query-{index}"},
            rationale=f"search {index}",
        )
        for index in range(MAX_ITERATIONS + 2)
    ]

    with patch(
        "graphs.nodes.agent_loop._plan_next_step_sync",
        side_effect=planner_steps,
    ):
        result = await agent_loop(
            _base_state(),
            kapruka_service=mock_service,
            client_ip=_CLIENT_IP,
        )

    assert result["agent_loop_done"] is True
    assert len(result["tool_trace"]) == MAX_ITERATIONS
    assert result["tool_call_count"] == MAX_ITERATIONS
    assert mock_service.search_products.call_count == MAX_ITERATIONS


@pytest.mark.asyncio
async def test_agent_loop_duplicate_tool_guard_forces_finish() -> None:
    """Duplicate tool+args skips re-invocation and forces finish on the next iteration."""
    mock_service = _mock_kapruka_service()
    search_args = {"q": "birthday cake", "currency": "LKR"}
    planner_steps = [
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "birthday cake"},
            rationale="initial search",
        ),
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "birthday cake"},
            rationale="duplicate search",
        ),
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "should not run"},
            rationale="never reached",
        ),
    ]

    with patch(
        "graphs.nodes.agent_loop._plan_next_step_sync",
        side_effect=planner_steps,
    ):
        result = await agent_loop(
            _base_state(),
            kapruka_service=mock_service,
            client_ip=_CLIENT_IP,
        )

    assert result["agent_loop_done"] is True
    assert len(result["tool_trace"]) == 1
    assert result["tool_trace"][0]["args"] == search_args
    assert mock_service.search_products.call_count == 1


@pytest.mark.asyncio
async def test_agent_loop_emits_status_events() -> None:
    """Status events are emitted at loop entry and per tool invocation."""
    mock_service = _mock_kapruka_service()
    mock_writer = MagicMock()
    planner_steps = [
        AgentPlannerStep(
            action="call_tool",
            tool_name=CHECK_DELIVERY_TOOL,
            tool_args={"city": "Kandy", "date": "2026-06-15"},
            rationale="check delivery",
        ),
        AgentPlannerStep(action="finish", rationale="done"),
    ]

    with (
        patch("graphs.nodes.agent_loop.get_stream_writer", return_value=mock_writer),
        patch(
            "graphs.nodes.agent_loop._plan_next_step_sync",
            side_effect=planner_steps,
        ),
    ):
        await agent_loop(
            _base_state(),
            kapruka_service=mock_service,
            client_ip=_CLIENT_IP,
        )

    status_messages = [
        call.args[0]["message"]
        for call in mock_writer.call_args_list
        if call.args and call.args[0].get("type") == "status"
    ]
    assert "Searching catalog…" in status_messages
    assert "Checking delivery…" in status_messages


@pytest.mark.asyncio
async def test_agent_loop_planner_uses_flash_model_only() -> None:
    """Planner never escalates to Pro — always gemini-2.5-flash."""
    mock_service = _mock_kapruka_service()
    mock_response = MagicMock()
    mock_response.parsed = AgentPlannerStep(action="finish", rationale="done")

    with patch(
        "graphs.nodes.agent_loop.generate_content_with_fallback",
        return_value=mock_response,
    ) as mock_generate:
        await agent_loop(
            _base_state(),
            kapruka_service=mock_service,
            client_ip=_CLIENT_IP,
            genai_client=MagicMock(),
        )

    assert mock_generate.call_args.kwargs["model"] == FLASH_MODEL
    assert mock_generate.call_args.kwargs["model"] != "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_agent_loop_requires_kapruka_service() -> None:
    with pytest.raises(ValueError, match="kapruka_service is required"):
        await agent_loop(_base_state())
