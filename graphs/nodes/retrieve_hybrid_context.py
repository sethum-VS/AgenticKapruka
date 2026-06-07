"""HybridRAG context retrieval — Zep preferences now; full Neo4j in PRD-047."""

from __future__ import annotations

import logging
from typing import Any, Literal

from graphs.state import AgentState, Intent
from lib.zep.client import ZepClient
from lib.zep.preferences import (
    extract_preferences,
    merge_preferences_into_hybrid_context,
    parse_preferences_from_facts,
)

logger = logging.getLogger(__name__)

RouteAfterAnalyzeIntent = Literal["retrieve_hybrid_context", "call_mcp_tools"]

_INTENTS_SKIP_HYBRID_CONTEXT: frozenset[Intent] = frozenset({"tracking", "checkout"})

_VALID_CURRENCIES: frozenset[str] = frozenset({"LKR", "USD", "GBP", "AUD", "CAD", "EUR"})


def route_after_analyze_intent(state: AgentState) -> RouteAfterAnalyzeIntent:
    """Conditional edge after analyze_intent: skip HybridRAG for tracking/checkout."""
    intent = state.get("intent")
    if intent in _INTENTS_SKIP_HYBRID_CONTEXT:
        logger.debug("route_after_analyze_intent: skipping hybrid context for %s", intent)
        return "call_mcp_tools"
    return "retrieve_hybrid_context"


async def retrieve_hybrid_context(
    state: AgentState,
    *,
    zep_client: ZepClient | None = None,
) -> dict[str, Any]:
    """LangGraph node: merge Zep preference hints into hybrid_context for MCP search."""
    thread_id = state.get("zep_thread_id")
    preferences: dict[str, str] = {}

    if thread_id and zep_client is not None:
        preferences = await extract_preferences(zep_client, thread_id)
    elif facts := state.get("zep_memory_facts"):
        preferences = parse_preferences_from_facts(facts)

    hybrid_context = merge_preferences_into_hybrid_context(
        state.get("hybrid_context"),
        preferences,
    )

    updates: dict[str, Any] = {"hybrid_context": hybrid_context}
    currency_hint = preferences.get("currency")
    if currency_hint and currency_hint in _VALID_CURRENCIES and state.get("currency") is None:
        updates["currency"] = currency_hint

    logger.debug(
        "retrieve_hybrid_context: preferences=%s hybrid_context=%s",
        preferences,
        hybrid_context,
    )
    return updates
