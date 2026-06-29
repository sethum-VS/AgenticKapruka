"""HybridRAG context retrieval — Neo4j vector search + traversal and Zep preferences."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from app.config import get_settings
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.chat.intent_heuristics import is_budget_refinement_message
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.routing import RouteAfterAnalyzeIntent, route_after_analyze_intent
from lib.embeddings.reranker import CrossEncoderService, get_reranker
from lib.embeddings.vertex_embeddings import embed_texts
from lib.neo4j.client import Neo4jClient
from lib.redis.client import RedisClient
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
from lib.neo4j.vector_search import VectorSearchHit, occasion_vector_search, vector_search
from lib.zep.client import ZepClient
from lib.zep.preferences import (
    extract_preferences,
    merge_preferences_into_hybrid_context,
    parse_preferences_from_facts,
)

logger = logging.getLogger(__name__)

_VALID_CURRENCIES: frozenset[str] = frozenset({"LKR", "USD", "GBP", "AUD", "CAD", "EUR"})

_VECTOR_SEARCH_TOP_K = 5
_GRAPH_TRAVERSE_MAX_HOPS = 2

EmbedTextsFn = Callable[[list[str]], Awaitable[list[list[float]]]]


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
    category_result, occasion_result = await asyncio.gather(
        vector_search(neo4j_client, query_embedding, top_k=_VECTOR_SEARCH_TOP_K),
        occasion_vector_search(neo4j_client, query_embedding, top_k=_VECTOR_SEARCH_TOP_K),
        return_exceptions=True,
    )
    category_hits: list[VectorSearchHit] = []
    occasion_hits: list[VectorSearchHit] = []
    if isinstance(category_result, BaseException):
        logger.warning(
            "retrieve_hybrid_context: category vector search failed; continuing with occasion only",
            exc_info=category_result,
        )
    else:
        category_hits = category_result
    if isinstance(occasion_result, BaseException):
        logger.warning(
            "retrieve_hybrid_context: occasion vector search failed; continuing with category only",
            exc_info=occasion_result,
        )
    else:
        occasion_hits = occasion_result
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
    redis_client: RedisClient | None = None,
    embed_fn: EmbedTextsFn | None = None,
) -> dict[str, Any]:
    """LangGraph node: GraphRAG + Zep preference hints merged into hybrid_context."""
    thread_id = state.get("zep_thread_id")
    preferences: dict[str, str] = {}

    hybrid_context: dict[str, Any] = dict(state.get("hybrid_context") or {})
    user_message = _extract_latest_user_message(state.get("messages") or [])
    intent_metadata: IntentMetadata | None = state.get("intent_metadata")
    topic_pivot = bool(
        intent_metadata
        and (
            intent_metadata.get("topic_pivot")
            or intent_metadata.get("discovery_context_reset")
        )
    )
    skip_graph_reembed = bool(
        is_budget_refinement_message(user_message) and hybrid_context,
    )
    if topic_pivot:
        hybrid_context = {}

    zep_task: asyncio.Task[dict[str, str]] | None = None
    if thread_id and zep_client is not None:
        zep_task = asyncio.create_task(extract_preferences(zep_client, thread_id))

    graph_task: asyncio.Task[dict[str, Any]] | None = None
    if neo4j_client is not None and not skip_graph_reembed:
        embed = embed_fn
        if embed is None:
            async def _embed_with_cache(texts: list[str]) -> list[list[float]]:
                return await embed_texts(texts, redis_client=redis_client)

            embed = _embed_with_cache
        graph_task = asyncio.create_task(
            _fetch_graph_hybrid_context(
                user_message,
                neo4j_client=neo4j_client,
                embed_fn=embed,
            ),
        )

    async_tasks: list[asyncio.Task[Any]] = []
    task_kinds: list[str] = []
    if zep_task is not None:
        async_tasks.append(zep_task)
        task_kinds.append("zep")
    if graph_task is not None:
        async_tasks.append(graph_task)
        task_kinds.append("graph")

    graph_context: dict[str, Any] = {}
    if async_tasks:
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        for kind, result in zip(task_kinds, results, strict=True):
            if kind == "zep":
                if isinstance(result, BaseException):
                    logger.warning(
                        "retrieve_hybrid_context: Zep preference fetch failed; continuing without",
                        exc_info=result,
                    )
                    facts = state.get("zep_memory_facts")
                    preferences = (
                        parse_preferences_from_facts(facts) if facts else {}
                    )
                else:
                    preferences = result
            elif kind == "graph":
                if isinstance(result, BaseException):
                    logger.exception(
                        "retrieve_hybrid_context: Neo4j GraphRAG failed; continuing with Zep only",
                        exc_info=result,
                    )
                elif result:
                    graph_context = result

    if zep_task is None:
        if facts := state.get("zep_memory_facts"):
            preferences = parse_preferences_from_facts(facts)

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
    elif graph_task is not None and neo4j_client is not None and not skip_graph_reembed:
        logger.warning(
            "retrieve_hybrid_context: Neo4j graph returned 0 products for %r — "
            "run `python scripts/bootstrap_neo4j.py` before local eval",
            user_message[:80],
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
    elif graph_context:
        graph_hints = graph_context.get("hints") or {}
        if graph_context.get("vector_hits") and not graph_hints:
            logger.debug(
                "retrieve_hybrid_context: rerank_prune_empty_hints query=%r vector_hits=%d",
                user_message[:80],
                len(graph_context.get("vector_hits") or []),
            )
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
