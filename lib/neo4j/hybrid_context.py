"""Build hybrid_context graph payloads from vector search and traversal."""

from __future__ import annotations

from typing import Any

from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import LABEL_CATEGORY, LABEL_OCCASION, REL_OCCASION_TO_CATEGORY
from lib.neo4j.traverse import TraversalNode, TraversalResult
from lib.neo4j.vector_search import VectorSearchHit

_FETCH_CATEGORY_DISPLAY_NAMES_CYPHER = f"""
MATCH (c:{LABEL_CATEGORY})
WHERE c.id IN $category_ids
RETURN c.id AS id, c.display_name AS display_name
""".strip()

_FETCH_CATEGORIES_FOR_OCCASIONS_CYPHER = f"""
MATCH (o:{LABEL_OCCASION})-[:{REL_OCCASION_TO_CATEGORY}]->(c:{LABEL_CATEGORY})
WHERE o.id IN $occasion_ids
RETURN DISTINCT c.id AS id
""".strip()

# Minimum vector-search score before a direct occasion hit becomes hints.occasion.
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


async def fetch_category_ids_for_occasions(
    client: Neo4jClient,
    occasion_ids: list[str],
) -> list[str]:
    """Fast-hop OCCASION_TO_CATEGORY for high-confidence occasion vector hits."""
    if not occasion_ids:
        return []
    rows = await client.execute(
        _FETCH_CATEGORIES_FOR_OCCASIONS_CYPHER,
        {"occasion_ids": occasion_ids},
    )
    return [str(row["id"]) for row in rows]


def _occasion_display_name(occasion_id: str, occasions: list[TraversalNode]) -> str:
    for node in occasions:
        if node.id == occasion_id:
            return node.display_name
    slug = occasion_id.rsplit(":", maxsplit=1)[-1]
    return slug.replace("-", " ").title()


def _best_occasion_hint(
    direct_occasion_hits: list[VectorSearchHit],
    occasions: list[TraversalNode],
) -> str | None:
    """Pick occasion hint from vector hits above threshold, else highest-weight traversal."""
    sorted_hits = sorted(direct_occasion_hits, key=lambda hit: hit.score, reverse=True)
    if sorted_hits and sorted_hits[0].score >= VECTOR_CONFIDENCE_THRESHOLD:
        return _occasion_display_name(sorted_hits[0].id, occasions)

    if not occasions:
        return None
    best = max(occasions, key=lambda node: node.weight)
    return best.display_name


def build_graph_hybrid_context(
    query: str,
    *,
    vector_hits: list[VectorSearchHit],
    display_names: dict[str, str],
    traversal: TraversalResult,
    direct_occasion_hits: list[VectorSearchHit] | None = None,
) -> dict[str, Any]:
    """Assemble graph-derived hybrid_context with MCP filter hints."""
    if not vector_hits and not (direct_occasion_hits or []):
        return {}

    ranked_hits = [
        {
            "id": hit.id,
            "score": hit.score,
            "display_name": display_names.get(hit.id, hit.id),
        }
        for hit in vector_hits
    ]
    ranked_occasion_hits = [
        {"id": hit.id, "score": hit.score} for hit in (direct_occasion_hits or [])
    ]
    top_category_id = vector_hits[0].id if vector_hits else next(iter(display_names), "")
    top_category = str(display_names.get(top_category_id, top_category_id))
    occasion_hint = _best_occasion_hint(direct_occasion_hits or [], traversal.occasions)

    hints: dict[str, str] = {}
    if top_category:
        hints["category"] = top_category
    if occasion_hint:
        hints["occasion"] = occasion_hint

    return {
        "vector_hits": ranked_hits,
        "direct_occasion_hits": ranked_occasion_hits,
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
    """True when occasion text already appears in the user query."""
    occasion_lower = occasion.lower().strip()
    if not occasion_lower:
        return True
    return occasion_lower in query.lower()


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
