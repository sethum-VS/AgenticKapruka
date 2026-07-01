"""Integration regressions for master flow supervisor."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google import genai
from langchain_core.messages import HumanMessage

from graphs.nodes.master_flow import master_flow
from graphs.shopping_graph import ShoppingGraphDeps, build_shopping_graph
from lib.chat.master_flow import MasterFlowAlignment


def _genai_client() -> MagicMock:
    return MagicMock(spec=genai.Client)


@pytest.mark.asyncio
async def test_long_session_budget_drift_invokes_master_flow() -> None:
    messages = [HumanMessage(content=f"turn {i}") for i in range(8)]
    messages.append(HumanMessage(content="gift for wife under 5000 rupees"))
    state: dict[str, Any] = {
        "messages": messages,
        "intent": "discovery",
        "last_visible_products": [{"id": "stale", "name": "Old Hamper"}],
        "session_search_query": "hamper",
    }
    with patch("lib.chat.master_flow.generate_content_with_fallback") as mock_llm:
        mock_llm.return_value = MagicMock(
            parsed=MasterFlowAlignment(
                decision="pivot",
                confidence=0.9,
                active_flow="carousel_context",
                context_reset=True,
            ),
            text="",
        )
        updates = await master_flow(state, genai_client=_genai_client())  # type: ignore[arg-type]
    assert updates.get("master_flow_invoked") is True
    assert updates.get("last_visible_products") is None


@pytest.mark.asyncio
async def test_delivery_only_clarify_short_circuit() -> None:
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="delivery fee to Colombo on 2026-07-05")],
        "intent": "discovery",
        "intent_metadata": {
            "requires_delivery_validation": True,
            "target_city": "Colombo",
            "delivery_date": "2026-07-05",
        },
        "last_visible_products": [{"id": "prior"}],
    }
    with patch("lib.chat.master_flow.generate_content_with_fallback") as mock_llm:
        mock_llm.return_value = MagicMock(
            parsed=MasterFlowAlignment(
                decision="redirect",
                confidence=0.85,
                active_flow="delivery_resolution",
                resolved_intent="general",
            ),
            text="",
        )
        updates = await master_flow(state, genai_client=_genai_client())  # type: ignore[arg-type]
    assert updates.get("master_flow_invoked") is True


@pytest.mark.asyncio
async def test_awaiting_date_unrelated_product_clarifies() -> None:
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="show me chocolate cakes")],
        "intent": "discovery",
        "session_awaiting_delivery_date": True,
    }
    with patch("lib.chat.master_flow.generate_content_with_fallback") as mock_llm:
        mock_llm.return_value = MagicMock(
            parsed=MasterFlowAlignment(
                decision="clarify",
                confidence=0.88,
                active_flow="awaiting_delivery_date",
                clarifying_question="Before we browse cakes — which delivery date do you need?",
            ),
            text="",
        )
        updates = await master_flow(state, genai_client=_genai_client())  # type: ignore[arg-type]
    assert "Before we browse" in (updates.get("master_clarifying_question") or "")


@pytest.mark.asyncio
async def test_checkout_interrupt_pauses() -> None:
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="what's the weather in Colombo?")],
        "intent": "general",
        "checkout_state": "delivery_city",
    }
    with patch("lib.chat.master_flow.generate_content_with_fallback") as mock_llm:
        mock_llm.return_value = MagicMock(
            parsed=MasterFlowAlignment(
                decision="redirect",
                confidence=0.9,
                active_flow="checkout_active",
                checkout_action="pause",
            ),
            text="",
        )
        updates = await master_flow(state, genai_client=_genai_client())  # type: ignore[arg-type]
    assert updates.get("checkout_paused") is True


@pytest.mark.asyncio
async def test_graph_master_flow_node_in_path() -> None:
    graph = build_shopping_graph(deps=ShoppingGraphDeps())
    node_names = set(graph.get_graph().nodes.keys())
    assert "master_flow" in node_names
