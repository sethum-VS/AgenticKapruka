"""HybridRAG context retrieval — Neo4j vector search + traversal and Zep preferences."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from app.config import get_settings
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, Intent
from lib.chat.intent_heuristics import is_budget_refinement_message
from lib.chat.intent_metadata import IntentMetadata
from lib.debug.trace import trace_route_decision
from lib.embeddings.reranker import CrossEncoderService, get_reranker
from lib.embeddings.vertex_embeddings import embed_texts
from lib.kapruka.product_id import contains_product_id
from lib.neo4j.client import Neo4jClient
from lib.neo4j.hybrid_context import (
    VECTOR_CONFIDENCE_THRESHOLD,
    build_graph_hybrid_context,
    enrich_anniversary_hints,
    enrich_birthday_cake_hints,
    enrich_chocolate_focus_hints,
    enrich_flower_fruit_negative_hints,
    fetch_category_display_names,
    fetch_category_ids_for_occasions,
)
from lib.neo4j.traverse import traverse_from_categories
from lib.neo4j.vector_search import occasion_vector_search, vector_search
from lib.zep.client import ZepClient
from lib.zep.preferences import (
    extract_preferences,
    merge_preferences_into_hybrid_context,
    parse_preferences_from_facts,
)

logger = logging.getLogger(__name__)

RouteAfterAnalyzeIntent = Literal[
    "retrieve_hybrid_context",
    "call_mcp_tools",
    "run_checkout_graph",
    "resolve_cart_product",
    "resolve_delivery_context",
    "generate_response",
]

_INTENTS_SKIP_HYBRID_CONTEXT: frozenset[Intent] = frozenset({"tracking"})

_VALID_CURRENCIES: frozenset[str] = frozenset({"LKR", "USD", "GBP", "AUD", "CAD", "EUR"})

_VECTOR_SEARCH_TOP_K = 5
_GRAPH_TRAVERSE_MAX_HOPS = 2

EmbedTextsFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def route_after_analyze_intent(state: AgentState) -> RouteAfterAnalyzeIntent:
    """Conditional edge after analyze_intent: checkout sub-graph or HybridRAG skip."""
    clarifying = state.get("agent_clarifying_question")
    if isinstance(clarifying, str) and clarifying.strip():
        trace_route_decision(
            from_node="analyze_intent",
            target="generate_response",
            intent=state.get("intent"),
            reason="clarifying question from intent preprocessing",
        )
        return "generate_response"

    intent_metadata: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
    if intent_metadata.get("support_topic"):
        trace_route_decision(
            from_node="analyze_intent",
            target="generate_response",
            intent=state.get("intent"),
            reason="support or policy FAQ",
        )
        return "generate_response"

    intent = state.get("intent")
    if intent == "checkout":
        logger.debug("route_after_analyze_intent: routing to checkout sub-graph")
        trace_route_decision(
            from_node="analyze_intent",
            target="run_checkout_graph",
            intent=intent,
            reason="checkout intent",
        )
        return "run_checkout_graph"
    if intent == "cart":
        logger.debug("route_after_analyze_intent: routing to cart transaction path")
        trace_route_decision(
            from_node="analyze_intent",
            target="resolve_cart_product",
            intent=intent,
            reason="cart add intent",
        )
        return "resolve_cart_product"
    if intent in _INTENTS_SKIP_HYBRID_CONTEXT:
        logger.debug("route_after_analyze_intent: skipping hybrid context for %s", intent)
        trace_route_decision(
            from_node="analyze_intent",
            target="call_mcp_tools",
            intent=intent,
            reason="intent skips hybrid context retrieval",
        )
        return "call_mcp_tools"
    if intent in ("discovery", "general"):
        user_message = _extract_latest_user_message(state.get("messages") or [])
        if contains_product_id(user_message):
            intent_metadata: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
            has_city = bool(intent_metadata.get("target_city")) or bool(
                intent_metadata.get("requires_delivery_validation"),
            )
            if has_city:
                logger.debug(
                    "route_after_analyze_intent: product ID + city → resolve_delivery_context",
                )
                trace_route_decision(
                    from_node="analyze_intent",
                    target="resolve_delivery_context",
                    intent=intent,
                    reason="product ID with delivery city",
                )
                return "resolve_delivery_context"
            logger.debug(
                "route_after_analyze_intent: product ID fast-path for intent=%s",
                intent,
            )
            trace_route_decision(
                from_node="analyze_intent",
                target="call_mcp_tools",
                intent=intent,
                reason="product ID in message",
            )
            return "call_mcp_tools"
    trace_route_decision(
        from_node="analyze_intent",
        target="retrieve_hybrid_context",
        intent=intent,
        reason="discovery/general path",
    )
    return "retrieve_hybrid_context"


def _dedupe_preserve_order(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


async def _fetch_graph_hybrid_context(
    query: str,
    *,
    neo4j_client: Neo4jClient,
    embed_fn: EmbedTextsFn,
    reranker: CrossEncoderService | None = None,
    reranker_threshold: float | None = None,
) -> dict[str, Any]:
    """Embed query, parallel Category/Occasion vector search, seed traversal, build hints."""
    stripped = query.strip()
    if not stripped:
        return {}

    embeddings = await embed_fn([stripped])
    if not embeddings:
        return {}

    query_embedding = embeddings[0]
    category_hits, occasion_hits = await asyncio.gather(
        vector_search(neo4j_client, query_embedding, top_k=_VECTOR_SEARCH_TOP_K),
        occasion_vector_search(neo4j_client, query_embedding, top_k=_VECTOR_SEARCH_TOP_K),
    )
    if not category_hits and not occasion_hits:
        return {}

    direct_category_ids = [hit.id for hit in category_hits]
    high_confidence_occasion_ids = [
        hit.id for hit in occasion_hits if hit.score >= VECTOR_CONFIDENCE_THRESHOLD
    ]
    occasion_category_ids = await fetch_category_ids_for_occasions(
        neo4j_client,
        high_confidence_occasion_ids,
    )
    seed_category_ids = _dedupe_preserve_order([*direct_category_ids, *occasion_category_ids])
    if not seed_category_ids:
        return {}

    display_names = await fetch_category_display_names(neo4j_client, seed_category_ids)
    traversal = await traverse_from_categories(
        neo4j_client,
        seed_category_ids,
        max_hops=_GRAPH_TRAVERSE_MAX_HOPS,
    )
    ranker = reranker if reranker is not None else get_reranker()
    threshold = (
        reranker_threshold if reranker_threshold is not None else get_settings().reranker_threshold
    )
    return build_graph_hybrid_context(
        stripped,
        vector_hits=category_hits,
        direct_occasion_hits=occasion_hits,
        display_names=display_names,
        traversal=traversal,
        reranker=ranker,
        reranker_threshold=threshold,
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
    intent_metadata: IntentMetadata | None = state.get("intent_metadata")
    topic_pivot = bool(intent_metadata and intent_metadata.get("topic_pivot"))
    skip_graph_reembed = bool(
        is_budget_refinement_message(user_message) and hybrid_context,
    )
    if topic_pivot:
        hybrid_context = {}

    if neo4j_client is not None and not skip_graph_reembed:
        embed = embed_fn or embed_texts
        try:
            graph_context = await _fetch_graph_hybrid_context(
                user_message,
                neo4j_client=neo4j_client,
                embed_fn=embed,
            )
            if graph_context:
                hybrid_context = _merge_graph_hybrid_context(
                    hybrid_context,
                    graph_context,
                    topic_pivot=topic_pivot,
                )
                product_count = len(graph_context.get("vector_hits") or []) + len(
                    graph_context.get("categories") or [],
                )
                if product_count == 0:
                    logger.warning(
                        "retrieve_hybrid_context: Neo4j graph returned 0 products for %r — "
                        "run `python scripts/bootstrap_neo4j.py` for local GraphRAG",
                        user_message[:80],
                    )
            else:
                logger.warning(
                    "retrieve_hybrid_context: Neo4j graph returned 0 products for %r — "
                    "run `python scripts/bootstrap_neo4j.py` before local eval",
                    user_message[:80],
                )
        except Exception:
            logger.exception(
                "retrieve_hybrid_context: Neo4j GraphRAG failed; continuing with Zep only",
            )

    hybrid_context = merge_preferences_into_hybrid_context(
        hybrid_context,
        preferences,
        user_message=user_message,
        topic_pivot=topic_pivot,
    )
    hybrid_context = enrich_flower_fruit_negative_hints(user_message, hybrid_context)
    hybrid_context = enrich_chocolate_focus_hints(
        user_message,
        hybrid_context,
        session_product_focus=state.get("session_product_focus"),
    )
    hybrid_context = enrich_birthday_cake_hints(
        user_message,
        hybrid_context,
        intent_metadata=intent_metadata,
    )
    hybrid_context = enrich_anniversary_hints(user_message, hybrid_context)

    graph_degraded = False
    if neo4j_client is not None and not skip_graph_reembed:
        vector_hits = hybrid_context.get("vector_hits") or []
        categories = hybrid_context.get("categories") or []
        if not vector_hits and not categories:
            graph_degraded = True

    updates: dict[str, Any] = {"hybrid_context": hybrid_context}
    if graph_degraded:
        meta = dict(intent_metadata or state.get("intent_metadata") or {})
        meta["graph_degraded"] = True
        updates["intent_metadata"] = cast(IntentMetadata, meta)
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
    *,
    topic_pivot: bool = False,
) -> dict[str, Any]:
    """Merge graph retrieval fields; graph hints fill gaps when Zep hints are absent."""
    merged: dict[str, Any] = dict(base)
    graph_hints = dict(graph_context.get("hints") or {})
    hints: dict[str, str] = dict(merged.get("hints") or {})

    for key, value in graph_hints.items():
        if topic_pivot and key in ("occasion", "category"):
            hints[key] = value
        else:
            hints.setdefault(key, value)

    merged["hints"] = hints
    for key in (
        "vector_hits",
        "direct_occasion_hits",
        "occasions",
        "product_types",
        "categories",
    ):
        if key in graph_context:
            merged[key] = graph_context[key]
    return merged
