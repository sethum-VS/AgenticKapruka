"""Unit tests for graphs.nodes.retrieve_hybrid_context stub."""

from __future__ import annotations

import inspect
from pathlib import Path

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
async def test_retrieve_hybrid_context_returns_empty_dict() -> None:
    """Stub returns empty hybrid_context without external services."""
    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "intent": "discovery",
        "session_id": "sess-hybrid-001",
    }

    result = await retrieve_hybrid_context(state)

    assert result == {"hybrid_context": {}}


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
