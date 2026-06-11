"""Build hybrid_context graph payloads from vector search and traversal."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from lib.chat.intent_metadata import IntentMetadata
from lib.chat.model_router import select_rewrite_model
from lib.embeddings.reranker import CrossEncoderService
from lib.genai.fallback import generate_content_with_fallback
from lib.neo4j.client import Neo4jClient
from lib.neo4j.embed_ontology import build_embedding_text
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

# Minimum vector-search score before a direct occasion hit seeds category traversal.
VECTOR_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_RERANKER_THRESHOLD = 0.45

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_INSTRUCTION = (
    "You rewrite casual Kapruka gift-shopping messages into concise product search queries.\n\n"
    "Incorporate the occasion naturally "
    '(e.g. "cake for mom" + Birthday → "birthday cake for mom").\n'
    "Do not merely append the occasion word. Keep the query short (under 12 words), "
    "ecommerce-focused, and faithful to the user's intent. "
    "Return only the rewritten search string in JSON."
)


class RewrittenSearchQuery(BaseModel):
    """Structured Gemini response for occasion-aware search rewrite."""

    q: str


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


@dataclass(frozen=True, slots=True)
class _RerankTarget:
    """Occasion or Category node candidate for cross-encoder scoring."""

    kind: Literal["category", "occasion"]
    node_id: str
    display_name: str
    description: str | None
    in_traversal: bool


def _build_rerank_targets(
    *,
    traversal: TraversalResult,
    vector_hits: list[VectorSearchHit],
    direct_occasion_hits: list[VectorSearchHit],
    display_names: dict[str, str],
) -> list[_RerankTarget]:
    """Collect unique Occasion/Category nodes for query–node reranking."""
    targets: list[_RerankTarget] = []
    seen: set[tuple[str, str]] = set()

    def _add(
        *,
        kind: Literal["category", "occasion"],
        node_id: str,
        display_name: str,
        description: str | None,
        in_traversal: bool,
    ) -> None:
        key = (kind, node_id)
        if key in seen:
            return
        seen.add(key)
        targets.append(
            _RerankTarget(
                kind=kind,
                node_id=node_id,
                display_name=display_name,
                description=description,
                in_traversal=in_traversal,
            )
        )

    for node in traversal.categories:
        _add(
            kind="category",
            node_id=node.id,
            display_name=node.display_name,
            description=node.description,
            in_traversal=True,
        )
    for node in traversal.occasions:
        _add(
            kind="occasion",
            node_id=node.id,
            display_name=node.display_name,
            description=node.description,
            in_traversal=True,
        )
    for hit in vector_hits:
        _add(
            kind="category",
            node_id=hit.id,
            display_name=str(display_names.get(hit.id, hit.id)),
            description=None,
            in_traversal=False,
        )
    for hit in direct_occasion_hits:
        _add(
            kind="occasion",
            node_id=hit.id,
            display_name=_occasion_display_name(hit.id, traversal.occasions),
            description=None,
            in_traversal=False,
        )
    return targets


def rerank_and_prune_traversal(
    query: str,
    traversal: TraversalResult,
    *,
    vector_hits: list[VectorSearchHit],
    direct_occasion_hits: list[VectorSearchHit],
    display_names: dict[str, str],
    reranker: CrossEncoderService,
    threshold: float = DEFAULT_RERANKER_THRESHOLD,
) -> tuple[TraversalResult, str | None, str | None]:
    """Score Occasion/Category nodes, prune traversal below threshold, pick hints."""
    targets = _build_rerank_targets(
        traversal=traversal,
        vector_hits=vector_hits,
        direct_occasion_hits=direct_occasion_hits,
        display_names=display_names,
    )
    if not targets:
        return traversal, None, None

    texts = [
        build_embedding_text(
            display_name=target.display_name,
            description=target.description,
        )
        for target in targets
    ]
    scores = reranker.score_pairs(query, texts)
    scored = list(zip(targets, scores, strict=True))

    surviving_ids = {
        target.node_id for target, score in scored if score >= threshold and target.in_traversal
    }
    pruned_nodes = tuple(
        node
        for node in traversal.nodes
        if node.label not in (LABEL_OCCASION, LABEL_CATEGORY) or node.id in surviving_ids
    )
    pruned = TraversalResult(nodes=pruned_nodes)

    category_hint = _best_hint_by_rerank_score(scored, kind="category", threshold=threshold)
    occasion_hint = _best_hint_by_rerank_score(scored, kind="occasion", threshold=threshold)
    return pruned, category_hint, occasion_hint


def _best_hint_by_rerank_score(
    scored: list[tuple[_RerankTarget, float]],
    *,
    kind: Literal["category", "occasion"],
    threshold: float,
) -> str | None:
    passing = [
        (score, target.display_name)
        for target, score in scored
        if target.kind == kind and score >= threshold
    ]
    if not passing:
        return None
    return max(passing, key=lambda item: item[0])[1]


def build_graph_hybrid_context(
    query: str,
    *,
    vector_hits: list[VectorSearchHit],
    display_names: dict[str, str],
    traversal: TraversalResult,
    direct_occasion_hits: list[VectorSearchHit] | None = None,
    reranker: CrossEncoderService | None = None,
    reranker_threshold: float = DEFAULT_RERANKER_THRESHOLD,
) -> dict[str, Any]:
    """Assemble graph-derived hybrid_context with MCP filter hints."""
    if not vector_hits and not (direct_occasion_hits or []):
        return {}

    occasion_hits = direct_occasion_hits or []
    pruned_traversal = traversal
    category_hint: str | None = None
    occasion_hint: str | None = None

    if reranker is not None:
        pruned_traversal, category_hint, occasion_hint = rerank_and_prune_traversal(
            query,
            traversal,
            vector_hits=vector_hits,
            direct_occasion_hits=occasion_hits,
            display_names=display_names,
            reranker=reranker,
            threshold=reranker_threshold,
        )

    ranked_hits = [
        {
            "id": hit.id,
            "score": hit.score,
            "display_name": display_names.get(hit.id, hit.id),
        }
        for hit in vector_hits
    ]
    ranked_occasion_hits = [{"id": hit.id, "score": hit.score} for hit in occasion_hits]

    hints: dict[str, str] = {}
    if category_hint:
        hints["category"] = category_hint
    if occasion_hint:
        hints["occasion"] = occasion_hint

    return {
        "vector_hits": ranked_hits,
        "direct_occasion_hits": ranked_occasion_hits,
        "occasions": [_serialize_traversal_node(node) for node in pruned_traversal.occasions],
        "product_types": [
            _serialize_traversal_node(node) for node in pruned_traversal.product_types
        ],
        "categories": [_serialize_traversal_node(node) for node in pruned_traversal.categories],
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


def _occasion_terms_in_query(query: str, occasion: str) -> bool:
    """True when occasion text already appears in the user query."""
    occasion_lower = occasion.lower().strip()
    if not occasion_lower:
        return True
    return occasion_lower in query.lower()


def get_discovery_occasion_hint(hybrid_context: dict[str, Any] | None) -> str | None:
    """Return occasion hint from graph or Zep preferences, if any."""
    context = hybrid_context or {}
    hints = context.get("hints") or {}
    preferences = context.get("preferences") or {}
    occasion = hints.get("occasion") or preferences.get("past_occasion")
    if occasion and str(occasion).strip():
        return str(occasion).strip()
    return None


def occasion_rewrite_needed(query: str, occasion: str) -> bool:
    """True when occasion context should influence q via Gemini rewrite."""
    stripped = query.strip()
    if not stripped or not occasion.strip():
        return False
    return not _occasion_terms_in_query(stripped, occasion)


_PRICE_SORT_RE = re.compile(
    r"\b("
    r"lowest|cheapest|low\s*price|budget|affordable|best\s*deal|"
    r"price\s*asc|under\s*\d|less\s*than"
    r")\b",
    re.IGNORECASE,
)
_CATALOG_BROWSE_RE = re.compile(
    r"\b("
    r"list\s+of|show\s+me\s+(all|everything)|any\s+(?:items?|itmes)|items?\s+today|"
    r"something\s+cheap|what.*cheapest|browse|all\s+products?"
    r")\b",
    re.IGNORECASE,
)
_MAX_PRICE_RE = re.compile(
    r"\b(?:under|below|less\s+than|upto|up\s+to)\s*(?:rs\.?|lkr)?\s*(\d[\d,]*)\s*(?:rs|lkr)?\b",
    re.IGNORECASE,
)
_META_QUERY_TOKENS = frozenset(
    {
        "can",
        "could",
        "give",
        "list",
        "lowest",
        "price",
        "items",
        "today",
        "show",
        "something",
        "cheapest",
        "budget",
        "please",
        "find",
        "any",
        "all",
        "everything",
        "browse",
        "products",
        "product",
        "cheap",
        "affordable",
        "deals",
        "deal",
        "under",
        "less",
        "than",
        "rs",
        "lkr",
        "itmes",
        "you",
        "your",
        "the",
        "for",
        "and",
        "with",
    }
)
_CATEGORY_SEARCH_TERMS: dict[str, str] = {
    "chocolates": "chocolate",
    "flowers": "flower",
    "birthday": "cake",
    "cakes": "cake",
    "gifts": "cake",
    "gift": "cake",
    "food": "cake",
    "fruits": "fruit",
    "perfumes": "perfume",
    "jewellery": "jewelry",
}

# Neo4j Category nodes are Kapruka parent departments (e.g. Cakes, Flowers). They are not
# valid kapruka_search_products category filters — only occasion/subcategory names work
# (e.g. Birthday). Passing parent names yields empty MCP results.
_INVALID_MCP_CATEGORY_FILTERS = frozenset(
    {
        "cakes",
        "birthday cakes",
        "flowers",
        "gifts",
        "gift",
        "chocolates",
        "food",
        "fruits",
        "perfumes",
        "jewellery",
        "jewelry",
    }
)

# Sri Lankan delivery cities often appear in chat queries ("cake for mom in Colombo") but
# pollute Kapruka keyword search when passed verbatim as q.
_DELIVERY_CITY_NAMES = (
    r"Colombo(?:\s+\d{2})?",
    r"Kandy",
    r"Galle",
    r"Negombo",
    r"Jaffna",
    r"Matara",
)
_CITY_PATTERN = "|".join(_DELIVERY_CITY_NAMES)
_STRIP_IN_CITY_RE = re.compile(
    rf"\b(?:in|to|near|around|within)\s+({_CITY_PATTERN})\b",
    re.IGNORECASE,
)
_STRIP_TRAILING_CITY_RE = re.compile(
    rf"\b(?:in|to)\s+({_CITY_PATTERN})\s*$",
    re.IGNORECASE,
)


def _product_like_tokens(message: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9']+", message.lower())
        if len(token) >= 3 and token not in _META_QUERY_TOKENS
    ]


def _is_meta_catalog_query(message: str) -> bool:
    """True when the user asks to browse/sort the catalog without naming a product."""
    stripped = message.strip()
    if not stripped:
        return False
    if _CATALOG_BROWSE_RE.search(stripped):
        return True
    return bool(_PRICE_SORT_RE.search(stripped) and not _product_like_tokens(stripped))


def _extract_max_price(message: str) -> float | None:
    """Parse budget caps like 'under 2000rs' into a numeric max_price."""
    match = _MAX_PRICE_RE.search(message)
    if not match:
        return None
    digits = match.group(1).replace(",", "")
    try:
        value = float(digits)
    except ValueError:
        return None
    return value if value > 0 else None


def _product_search_keyword(token: str) -> str:
    """Map a user token to a Kapruka-friendly product keyword."""
    normalized = token.lower().strip()
    mapped = _CATEGORY_SEARCH_TERMS.get(normalized)
    if mapped:
        return mapped
    if normalized.endswith("s") and len(normalized) > 4:
        return normalized.rstrip("s")
    return normalized


def _fallback_search_query(category: str | None) -> str:
    """Pick a broad Kapruka keyword that returns purchasable products."""
    if category:
        normalized = category.lower().strip()
        mapped = _CATEGORY_SEARCH_TERMS.get(normalized)
        if mapped:
            return mapped
        first_word = normalized.split()[0]
        if len(first_word) >= 3:
            if first_word.endswith("s") and len(first_word) > 4:
                return first_word.rstrip("s")
            return first_word
    return "cake"


DISCOVERY_SEARCH_TOOL = "kapruka_search_products"
DISCOVERY_CHECK_DELIVERY_TOOL = "kapruka_check_delivery"


def requires_discovery_delivery_check(intent_metadata: IntentMetadata | None) -> bool:
    """True when preprocessing flagged a destination city needing MCP delivery validation."""
    if intent_metadata is None:
        return False
    return bool(intent_metadata.get("requires_delivery_validation")) and bool(
        intent_metadata.get("target_city"),
    )


def discovery_tool_manifest(intent_metadata: IntentMetadata | None) -> frozenset[str]:
    """Discovery-turn MCP tools bound from hybrid context and delivery metadata."""
    tools: set[str] = {DISCOVERY_SEARCH_TOOL}
    if requires_discovery_delivery_check(intent_metadata):
        tools.add(DISCOVERY_CHECK_DELIVERY_TOOL)
    return frozenset(tools)


def build_discovery_delivery_args(intent_metadata: IntentMetadata | None) -> dict[str, Any]:
    """Map intent_metadata city constraint to kapruka_check_delivery arguments."""
    if not requires_discovery_delivery_check(intent_metadata):
        return {}
    city = intent_metadata.get("target_city") if intent_metadata else None
    if not city:
        return {}
    return {"city": city}


def _is_valid_mcp_category_filter(name: str) -> bool:
    stripped = name.strip()
    if not stripped:
        return False
    return stripped.lower() not in _INVALID_MCP_CATEGORY_FILTERS


def _resolve_mcp_category_filter(hybrid_context: dict[str, Any] | None) -> str | None:
    """Pick a Kapruka MCP subcategory filter from graph/Zep hints.

    Graph reranker stores parent departments under ``hints['category']`` (Neo4j
    Category nodes) and browse subcategories under ``hints['occasion']`` (Neo4j
    Occasion nodes). Only occasion-level names are valid MCP category filters.
    """
    context = hybrid_context or {}
    hints = context.get("hints") or {}
    preferences = context.get("preferences") or {}

    for raw in (
        hints.get("occasion"),
        preferences.get("favorite_category"),
    ):
        if not raw:
            continue
        name = str(raw).strip()
        if _is_valid_mcp_category_filter(name):
            return name
    return None


def strip_location_from_search_query(
    query: str,
    intent_metadata: IntentMetadata | None = None,
) -> str:
    """Remove delivery city phrases from product search q.

    Cities belong in kapruka_check_delivery, not kapruka_search_products q.
    """
    stripped = query.strip()
    if not stripped:
        return stripped

    cleaned = _STRIP_IN_CITY_RE.sub("", stripped)
    cleaned = _STRIP_TRAILING_CITY_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.-")

    target_city = intent_metadata.get("target_city") if intent_metadata else None
    if target_city:
        city_pattern = re.compile(rf"\b{re.escape(target_city)}\b", re.IGNORECASE)
        cleaned = city_pattern.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.-")

    return cleaned or stripped


def build_discovery_search_args(
    user_message: str,
    hybrid_context: dict[str, Any] | None,
    *,
    currency: str,
    intent_metadata: IntentMetadata | None = None,
) -> dict[str, Any]:
    """Map graph/Zep hybrid_context hints to kapruka_search_products arguments."""
    context = hybrid_context or {}
    category = _resolve_mcp_category_filter(context)
    query = strip_location_from_search_query(user_message.strip(), intent_metadata)

    args: dict[str, Any] = {"q": query, "currency": currency}
    if category:
        args["category"] = category

    if _PRICE_SORT_RE.search(query):
        args["sort"] = "price_asc"

    max_price = _extract_max_price(query)
    if max_price is not None:
        args["max_price"] = max_price
        args["sort"] = "price_asc"
        product_tokens = _product_like_tokens(query)
        if product_tokens:
            args["q"] = _product_search_keyword(product_tokens[0])
        else:
            args["q"] = _fallback_search_query(category)

    if _is_meta_catalog_query(query):
        args["q"] = _fallback_search_query(category)

    return args


def _parse_rewrite_response(response: types.GenerateContentResponse) -> str:
    """Parse structured or JSON text rewrite from a Gemini response."""
    if response.parsed is not None:
        if isinstance(response.parsed, RewrittenSearchQuery):
            return response.parsed.q.strip()
        validated = RewrittenSearchQuery.model_validate(response.parsed)
        return validated.q.strip()

    raw_text = (response.text or "").strip()
    if not raw_text:
        msg = "Gemini returned empty search rewrite"
        raise ValueError(msg)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        msg = f"Gemini rewrite response is not valid JSON: {raw_text!r}"
        raise ValueError(msg) from exc

    try:
        return RewrittenSearchQuery.model_validate(payload).q.strip()
    except ValidationError as exc:
        msg = f"Gemini rewrite JSON failed validation: {payload!r}"
        raise ValueError(msg) from exc


def _rewrite_search_query_sync(
    client: genai.Client | None,
    user_message: str,
    occasion: str,
) -> str:
    """Blocking Gemini call; run via asyncio.to_thread from rewrite helper."""
    prompt = f"User message: {user_message}\nOccasion context: {occasion}"
    response = generate_content_with_fallback(
        client=client,
        model=select_rewrite_model(),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=REWRITE_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=RewrittenSearchQuery,
            temperature=0,
        ),
    )
    rewritten = _parse_rewrite_response(response)
    if not rewritten:
        msg = "Gemini returned empty rewritten search query"
        raise ValueError(msg)
    return rewritten


async def rewrite_search_query_with_occasion(
    user_message: str,
    occasion: str,
    *,
    genai_client: genai.Client | None = None,
) -> str:
    """Rewrite q with Gemini when occasion must influence search without naive concatenation."""
    stripped = user_message.strip()
    occasion_stripped = occasion.strip()
    if not stripped or not occasion_stripped:
        return stripped
    if not occasion_rewrite_needed(stripped, occasion_stripped):
        return stripped

    client = genai_client
    try:
        return await asyncio.to_thread(
            _rewrite_search_query_sync,
            client,
            stripped,
            occasion_stripped,
        )
    except Exception:
        logger.warning(
            "rewrite_search_query_with_occasion failed; using raw user message",
            exc_info=True,
        )
        return stripped
