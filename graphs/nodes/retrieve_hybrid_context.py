"""HybridRAG context retrieval — Neo4j vector search + traversal and Zep preferences."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, Intent
from lib.embeddings.vertex_embeddings import embed_texts
from lib.neo4j.client import Neo4jClient
from lib.neo4j.hybrid_context import (
    build_graph_hybrid_context,
    fetch_category_display_names,
)
from lib.neo4j.traverse import traverse_from_categories
from lib.neo4j.vector_search import vector_search
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

_VECTOR_SEARCH_TOP_K = 5
_GRAPH_TRAVERSE_MAX_HOPS = 2

EmbedTextsFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def route_after_analyze_intent(state: AgentState) -> RouteAfterAnalyzeIntent:
    """Conditional edge after analyze_intent: skip HybridRAG for tracking/checkout."""
    intent = state.get("intent")
    if intent in _INTENTS_SKIP_HYBRID_CONTEXT:
        logger.debug("route_after_analyze_intent: skipping hybrid context for %s", intent)
        return "call_mcp_tools"
    return "retrieve_hybrid_context"


async def _fetch_graph_hybrid_context(
    query: str,
    *,
    neo4j_client: Neo4jClient,
    embed_fn: EmbedTextsFn,
) -> dict[str, Any]:
    """Embed query, vector-search categories, traverse ontology, build MCP hints."""
    stripped = query.strip()
    if not stripped:
        return {}

    embeddings = await embed_fn([stripped])
    if not embeddings:
        return {}

    hits = await vector_search(neo4j_client, embeddings[0], top_k=_VECTOR_SEARCH_TOP_K)
    if not hits:
        return {}

    category_ids = [hit.id for hit in hits]
    display_names = await fetch_category_display_names(neo4j_client, category_ids)
    traversal = await traverse_from_categories(
        neo4j_client,
        category_ids,
        max_hops=_GRAPH_TRAVERSE_MAX_HOPS,
    )
    return build_graph_hybrid_context(
        stripped,
        vector_hits=hits,
        display_names=display_names,
        traversal=traversal,
    )


async def retrieve_hybrid_context(
    state: AgentState,
    *,
    zep_client: ZepClient | None = None,
    neo4j_client: Neo4jClient | None = None,
    embed_fn: EmbedTextsFn | None = None,
) -> dict[str, Any]:
    """LangGraph node: GraphRAG + Zep preference hints merged into hybrid_context."""
    thread_id = state.get("zep_thread_id")
    preferences: dict[str, str] = {}

    if thread_id and zep_client is not None:
        preferences = await extract_preferences(zep_client, thread_id)
    elif facts := state.get("zep_memory_facts"):
        preferences = parse_preferences_from_facts(facts)

    hybrid_context: dict[str, Any] = dict(state.get("hybrid_context") or {})
    user_message = _extract_latest_user_message(state.get("messages") or [])

    if neo4j_client is not None:
        embed = embed_fn or embed_texts
        try:
            graph_context = await _fetch_graph_hybrid_context(
                user_message,
                neo4j_client=neo4j_client,
                embed_fn=embed,
            )
            if graph_context:
                hybrid_context = _merge_graph_hybrid_context(hybrid_context, graph_context)
        except Exception:
            logger.exception(
                "retrieve_hybrid_context: Neo4j GraphRAG failed; continuing with Zep only",
            )

    hybrid_context = merge_preferences_into_hybrid_context(hybrid_context, preferences)

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


def _merge_graph_hybrid_context(
    base: dict[str, Any],
    graph_context: dict[str, Any],
) -> dict[str, Any]:
    """Merge graph retrieval fields; graph hints fill gaps when Zep hints are absent."""
    merged: dict[str, Any] = dict(base)
    graph_hints = dict(graph_context.get("hints") or {})
    hints: dict[str, str] = dict(merged.get("hints") or {})

    for key, value in graph_hints.items():
        hints.setdefault(key, value)

    merged["hints"] = hints
    for key in ("vector_hits", "occasions", "product_types", "categories"):
        if key in graph_context:
            merged[key] = graph_context[key]
    return merged
