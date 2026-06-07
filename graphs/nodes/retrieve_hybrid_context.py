"""HybridRAG context retrieval — stub until PRD-047 Neo4j implementation."""

from __future__ import annotations

import logging
from typing import Any, Literal

from graphs.state import AgentState, Intent

logger = logging.getLogger(__name__)

RouteAfterAnalyzeIntent = Literal["retrieve_hybrid_context", "call_mcp_tools"]

_INTENTS_SKIP_HYBRID_CONTEXT: frozenset[Intent] = frozenset({"tracking", "checkout"})


def route_after_analyze_intent(state: AgentState) -> RouteAfterAnalyzeIntent:
    """Conditional edge after analyze_intent: skip HybridRAG for tracking/checkout."""
    intent = state.get("intent")
    if intent in _INTENTS_SKIP_HYBRID_CONTEXT:
        logger.debug("route_after_analyze_intent: skipping hybrid context for %s", intent)
        return "call_mcp_tools"
    return "retrieve_hybrid_context"


async def retrieve_hybrid_context(state: AgentState) -> dict[str, Any]:
    """LangGraph node: stub HybridRAG context (full Neo4j path deferred to PRD-047)."""
    _ = state
    logger.debug("retrieve_hybrid_context: returning empty stub context")
    return {"hybrid_context": {}}
