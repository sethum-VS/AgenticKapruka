"""Build hybrid_context graph payloads from vector search and traversal."""

from __future__ import annotations

import re
from typing import Any

from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import LABEL_CATEGORY
from lib.neo4j.traverse import TraversalNode, TraversalResult
from lib.neo4j.vector_search import VectorSearchHit

_FETCH_CATEGORY_DISPLAY_NAMES_CYPHER = f"""
MATCH (c:{LABEL_CATEGORY})
WHERE c.id IN $category_ids
RETURN c.id AS id, c.display_name AS display_name
""".strip()

_WORD_RE = re.compile(r"[a-z0-9]+")

# Minimum top vector-search score before graph occasion hints augment MCP `q`.
VECTOR_CONFIDENCE_THRESHOLD = 0.65


async def fetch_category_display_names(
    client: Neo4jClient,
    category_ids: list[str],
) -> dict[str, str]:
    """Return category id → display_name for vector search hits."""
    if not category_ids:
        return {}
    rows = await client.execute(
        _FETCH_CATEGORY_DISPLAY_NAMES_CYPHER,
        {"category_ids": category_ids},
    )
    return {str(row["id"]): str(row.get("display_name") or row["id"]) for row in rows}


def _query_tokens(query: str) -> set[str]:
    return set(_WORD_RE.findall(query.lower()))


def _best_occasion_hint(query: str, occasions: list[TraversalNode]) -> str | None:
    if not occasions:
        return None
    tokens = _query_tokens(query)
    scored: list[tuple[int, float, str]] = []
    for occasion in occasions:
        name_tokens = _query_tokens(occasion.display_name)
        overlap = len(tokens & name_tokens)
        scored.append((overlap, occasion.weight, occasion.display_name))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_overlap, _, best_name = scored[0]
    if best_overlap > 0:
        return best_name
    return scored[0][2]


def build_graph_hybrid_context(
    query: str,
    *,
    vector_hits: list[VectorSearchHit],
    display_names: dict[str, str],
    traversal: TraversalResult,
) -> dict[str, Any]:
    """Assemble graph-derived hybrid_context with MCP filter hints."""
    if not vector_hits:
        return {}

    ranked_hits = [
        {
            "id": hit.id,
            "score": hit.score,
            "display_name": display_names.get(hit.id, hit.id),
        }
        for hit in vector_hits
    ]
    top_category = str(display_names.get(vector_hits[0].id, vector_hits[0].id))
    occasion_hint = _best_occasion_hint(query, traversal.occasions)

    hints: dict[str, str] = {"category": top_category}
    if occasion_hint:
        hints["occasion"] = occasion_hint

    return {
        "vector_hits": ranked_hits,
        "occasions": [_serialize_traversal_node(node) for node in traversal.occasions],
        "product_types": [_serialize_traversal_node(node) for node in traversal.product_types],
        "categories": [_serialize_traversal_node(node) for node in traversal.categories],
        "hints": hints,
    }


def _serialize_traversal_node(node: TraversalNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "display_name": node.display_name,
        "hop": node.hop,
        "relationship_type": node.relationship_type,
        "weight": node.weight,
        "seed_id": node.seed_id,
    }


def _top_vector_confidence(hybrid_context: dict[str, Any]) -> float | None:
    """Return the best vector-search score from hybrid_context, if present."""
    hits = hybrid_context.get("vector_hits") or []
    if not hits:
        return None
    score = hits[0].get("score")
    if score is None:
        return None
    return float(score)


def _occasion_terms_in_query(query: str, occasion: str) -> bool:
    """True when substantive occasion tokens already appear in the user query."""
    query_tokens = _query_tokens(query)
    occasion_tokens = _query_tokens(occasion)
    substantive = {token for token in occasion_tokens if len(token) > 2}
    if not substantive:
        substantive = occasion_tokens
    return bool(substantive & query_tokens)


def _augment_query_with_occasion(query: str, occasion: str) -> str:
    """Append occasion keywords when they are not already in the query."""
    stripped = query.strip()
    if not stripped or _occasion_terms_in_query(stripped, occasion):
        return stripped
    return f"{stripped} {occasion.strip()}".strip()


def build_discovery_search_args(
    user_message: str,
    hybrid_context: dict[str, Any] | None,
    *,
    currency: str,
) -> dict[str, Any]:
    """Map graph/Zep hybrid_context hints to kapruka_search_products arguments."""
    context = hybrid_context or {}
    hints = context.get("hints") or {}
    preferences = context.get("preferences") or {}

    category = hints.get("category") or preferences.get("favorite_category")
    occasion = hints.get("occasion") or preferences.get("past_occasion")

    query = user_message.strip()
    confidence = _top_vector_confidence(context)
    if occasion and confidence is not None and confidence >= VECTOR_CONFIDENCE_THRESHOLD:
        query = _augment_query_with_occasion(query, occasion)

    args: dict[str, Any] = {"q": query, "currency": currency}
    if category:
        args["category"] = category
    return args
