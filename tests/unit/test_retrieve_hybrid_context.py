"""Unit tests for graphs.nodes.retrieve_hybrid_context."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.retrieve_hybrid_context import (
    retrieve_hybrid_context,
    route_after_analyze_intent,
)
from graphs.state import AgentState, Intent


def test_retrieve_hybrid_context_module_has_no_neo4j_dependency() -> None:
    """Stub module must not import Neo4j (full implementation is PRD-047)."""
    module_path = inspect.getsourcefile(retrieve_hybrid_context)
    assert module_path is not None
    source = Path(module_path).read_text(encoding="utf-8")
    assert "from lib.neo4j" not in source
    assert "import neo4j" not in source


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_returns_empty_without_zep() -> None:
    """No Zep client or facts yields empty hybrid_context."""
    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "intent": "discovery",
        "session_id": "sess-hybrid-001",
    }

    result = await retrieve_hybrid_context(state)

    assert result == {"hybrid_context": {}}


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_merges_preferences_from_memory_facts() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="show me gifts")],
        "intent": "discovery",
        "session_id": "sess-hybrid-002",
        "zep_memory_facts": ["User prefers Birthday cakes"],
    }

    result = await retrieve_hybrid_context(state)

    assert result["hybrid_context"]["preferences"]["favorite_category"] == "Birthday"
    assert result["hybrid_context"]["hints"]["category"] == "Birthday"


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_injects_currency_from_preferences() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="show me gifts")],
        "intent": "discovery",
        "session_id": "sess-hybrid-003",
        "zep_memory_facts": ["User shops in USD"],
    }

    result = await retrieve_hybrid_context(state)

    assert result["currency"] == "USD"
    assert result["hybrid_context"]["hints"]["currency"] == "USD"


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_calls_extract_preferences_with_zep_client() -> None:
    zep_client = AsyncMock()
    with patch(
        "graphs.nodes.retrieve_hybrid_context.extract_preferences",
        new=AsyncMock(return_value={"favorite_category": "Flowers"}),
    ) as mock_extract:
        state: AgentState = {
            "messages": [],
            "intent": "discovery",
            "zep_thread_id": "thread-hybrid-004",
        }

        result = await retrieve_hybrid_context(state, zep_client=zep_client)

    mock_extract.assert_awaited_once_with(zep_client, "thread-hybrid-004")
    assert result["hybrid_context"]["hints"]["category"] == "Flowers"


@pytest.mark.parametrize(
    ("intent", "expected_route"),
    [
        ("discovery", "retrieve_hybrid_context"),
        ("general", "retrieve_hybrid_context"),
        ("tracking", "call_mcp_tools"),
        ("checkout", "call_mcp_tools"),
    ],
)
def test_route_after_analyze_intent_skips_tracking_and_checkout(
    intent: Intent,
    expected_route: str,
) -> None:
    state: AgentState = {
        "messages": [],
        "intent": intent,
        "session_id": "sess-route-001",
    }
    assert route_after_analyze_intent(state) == expected_route


def test_route_after_analyze_intent_defaults_to_retrieve_when_intent_missing() -> None:
    state: AgentState = {"messages": [], "session_id": "sess-route-002"}
    assert route_after_analyze_intent(state) == "retrieve_hybrid_context"
