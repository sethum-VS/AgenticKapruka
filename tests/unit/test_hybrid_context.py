"""Unit tests for graph hybrid_context assembly and occasion hint resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lib.embeddings.reranker import CrossEncoderService
from lib.neo4j.hybrid_context import (
    DEFAULT_RERANKER_THRESHOLD,
    RewrittenSearchQuery,
    build_discovery_delivery_args,
    build_discovery_search_args,
    build_graph_hybrid_context,
    discovery_tool_manifest,
    occasion_rewrite_needed,
    requires_discovery_delivery_check,
    rerank_and_prune_traversal,
    rewrite_search_query_with_occasion,
)
from lib.neo4j.ontology import LABEL_OCCASION
from lib.neo4j.traverse import TraversalNode, TraversalResult
from lib.neo4j.vector_search import VectorSearchHit


def _occasion_node(
    *,
    occasion_id: str,
    display_name: str,
    weight: float,
    description: str | None = None,
) -> TraversalNode:
    return TraversalNode(
        id=occasion_id,
        label=LABEL_OCCASION,
        display_name=display_name,
        description=description,
        hop=1,
        relationship_type="OCCASION_TO_CATEGORY",
        weight=weight,
        seed_id="category:flowers",
    )


class _SequenceReranker(CrossEncoderService):
    """Return predetermined scores in pair order for unit tests."""

    def __init__(self, scores: list[float]) -> None:
        super().__init__(model=None)  # type: ignore[arg-type]
        self._scores = scores

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        assert len(texts) == len(self._scores)
        return list(self._scores)


def test_hints_use_highest_reranker_score_not_vector_or_weight() -> None:
    """Cross-encoder score overrides vector similarity and traversal edge weight."""
    traversal = TraversalResult(
        nodes=(
            _occasion_node(
                occasion_id="occasion:birthday",
                display_name="Birthday",
                weight=2.0,
            ),
            _occasion_node(
                occasion_id="occasion:wedding",
                display_name="Wedding",
                weight=0.5,
            ),
        ),
    )
    direct_hits = [
        VectorSearchHit(id="occasion:wedding", score=0.99),
        VectorSearchHit(id="occasion:birthday", score=0.4),
    ]
    # targets: birthday traversal, wedding traversal, cakes vector (wedding direct deduped)
    reranker = _SequenceReranker([0.5, 0.9, 0.6])

    context = build_graph_hybrid_context(
        "cake for mom",
        vector_hits=[VectorSearchHit(id="category:cakes", score=0.1)],
        direct_occasion_hits=direct_hits,
        display_names={"category:cakes": "Cakes"},
        traversal=traversal,
        reranker=reranker,
        reranker_threshold=DEFAULT_RERANKER_THRESHOLD,
    )

    assert context["hints"]["occasion"] == "Wedding"
    assert context["hints"]["category"] == "Cakes"


def test_reranker_prunes_low_scoring_traversal_nodes() -> None:
    """Occasion/Category traversal nodes below RERANKER_THRESHOLD are dropped."""
    traversal = TraversalResult(
        nodes=(
            _occasion_node(
                occasion_id="occasion:anniversary",
                display_name="Anniversary",
                weight=1.2,
            ),
            _occasion_node(
                occasion_id="occasion:birthday",
                display_name="Birthday",
                weight=2.5,
            ),
        ),
    )
    reranker = _SequenceReranker([0.8, 0.2, 0.9])

    pruned, category_hint, occasion_hint = rerank_and_prune_traversal(
        "something elegant",
        traversal,
        vector_hits=[VectorSearchHit(id="category:flowers", score=0.7)],
        direct_occasion_hits=[],
        display_names={"category:flowers": "Flowers"},
        reranker=reranker,
        threshold=DEFAULT_RERANKER_THRESHOLD,
    )

    assert occasion_hint == "Anniversary"
    assert category_hint == "Flowers"
    assert len(pruned.occasions) == 1
    assert pruned.occasions[0].display_name == "Anniversary"


def test_reranker_omits_hints_below_threshold() -> None:
    """No hints when every Occasion/Category candidate scores below threshold."""
    reranker = _SequenceReranker([0.2, 0.1])

    context = build_graph_hybrid_context(
        "gift ideas",
        vector_hits=[VectorSearchHit(id="category:gifts", score=0.8)],
        direct_occasion_hits=[VectorSearchHit(id="occasion:wedding", score=0.3)],
        display_names={"category:gifts": "Gifts"},
        traversal=TraversalResult(nodes=()),
        reranker=reranker,
        reranker_threshold=DEFAULT_RERANKER_THRESHOLD,
    )

    assert context.get("hints") == {}
    assert context["vector_hits"][0]["id"] == "category:gifts"


def test_discovery_tool_manifest_includes_check_delivery_for_city_metadata() -> None:
    metadata = {
        "requires_delivery_validation": True,
        "target_city": "Kandy",
    }
    assert requires_discovery_delivery_check(metadata) is True
    assert discovery_tool_manifest(metadata) == frozenset(
        {"kapruka_search_products", "kapruka_check_delivery"},
    )
    assert build_discovery_delivery_args(metadata) == {"city": "Kandy"}


def test_discovery_tool_manifest_search_only_without_delivery_flag() -> None:
    metadata = {
        "requires_delivery_validation": False,
        "target_city": None,
    }
    assert requires_discovery_delivery_check(metadata) is False
    assert discovery_tool_manifest(metadata) == frozenset({"kapruka_search_products"})
    assert build_discovery_delivery_args(metadata) == {}


def test_build_discovery_search_args_preserves_raw_user_query() -> None:
    """kapruka_search_products q must stay the user's message without occasion concatenation."""
    args = build_discovery_search_args(
        "  cake for mom  ",
        {
            "hints": {"category": "Birthday", "occasion": "Birthday"},
            "vector_hits": [{"id": "category:cakes", "score": 0.91}],
        },
        currency="LKR",
    )

    assert args["q"] == "cake for mom"
    assert args["category"] == "Birthday"
    assert args["currency"] == "LKR"
    assert args["q"] != "cake for mom Birthday"


def test_build_discovery_search_args_ignores_parent_department_favorite_category() -> None:
    args = build_discovery_search_args(
        "something nice",
        {"preferences": {"favorite_category": "Flowers"}},
        currency="USD",
    )

    assert args["q"] == "something nice"
    assert "category" not in args
    assert args["currency"] == "USD"


def test_build_discovery_search_args_rewrites_meta_price_browse_query() -> None:
    """Catalog-style lowest-price requests map to a searchable q with price_asc sort."""
    args = build_discovery_search_args(
        "can u give me list of lowest price items today",
        {},
        currency="LKR",
    )

    assert args["q"] == "cake"
    assert args["sort"] == "price_asc"
    assert args["currency"] == "LKR"


def test_build_discovery_search_args_price_sort_preserves_product_query() -> None:
    """Explicit product terms stay in q while still applying price_asc sort."""
    args = build_discovery_search_args(
        "cheapest birthday cake",
        {"hints": {"occasion": "Birthday"}},
        currency="LKR",
    )

    assert args["q"] == "cheapest birthday cake"
    assert args["sort"] == "price_asc"
    assert args["category"] == "Birthday"


def test_build_discovery_search_args_ignores_parent_department_category_hint() -> None:
    """Graph Category hints like Cakes must not become MCP category filters."""
    args = build_discovery_search_args(
        "chocolates",
        {"hints": {"category": "Cakes"}},
        currency="LKR",
    )

    assert args["q"] == "chocolates"
    assert "category" not in args


def test_build_discovery_search_args_prefers_occasion_over_parent_category() -> None:
    args = build_discovery_search_args(
        "cake for mom",
        {"hints": {"category": "Cakes", "occasion": "Birthday"}},
        currency="LKR",
    )

    assert args["category"] == "Birthday"


def test_build_discovery_search_args_meta_browse_drops_parent_category_filter() -> None:
    args = build_discovery_search_args(
        "show me any items",
        {"hints": {"category": "Cakes"}},
        currency="LKR",
    )

    assert args["q"] == "cake"
    assert "category" not in args


def test_build_discovery_search_args_parses_under_price_budget() -> None:
    """Kapruka MCP rejects literal 'cakes under 2000rs'; map to q + max_price."""
    args = build_discovery_search_args(
        "cakes under 2000rs",
        {"hints": {"category": "Cakes", "occasion": "Birthday"}},
        currency="LKR",
    )

    assert args["q"] == "cake"
    assert args["max_price"] == 2000.0
    assert args["sort"] == "price_asc"
    assert args["category"] == "Birthday"


def test_build_discovery_search_args_strips_trailing_city_from_query() -> None:
    """Location in chat must not pollute Kapruka keyword search."""
    args = build_discovery_search_args(
        "Birthday cake for mom in Colombo",
        {"hints": {"occasion": "Birthday"}},
        currency="LKR",
    )

    assert args["q"] == "Birthday cake for mom"
    assert args["category"] == "Birthday"


def test_build_discovery_search_args_meta_browse_tolerates_itmes_typo() -> None:
    args = build_discovery_search_args(
        "show me any itmes",
        {"hints": {"category": "Cakes"}},
        currency="LKR",
    )

    assert args["q"] == "cake"
    assert "category" not in args


def test_occasion_rewrite_needed_when_terms_absent() -> None:
    assert occasion_rewrite_needed("cake for mom", "Birthday") is True
    assert occasion_rewrite_needed("birthday cake for mom", "Birthday") is False


@pytest.mark.asyncio
async def test_rewrite_search_query_with_occasion_uses_gemini() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = RewrittenSearchQuery(q="birthday cake for mom")
    mock_response.text = '{"q": "birthday cake for mom"}'
    mock_client.models.generate_content.return_value = mock_response

    with patch(
        "lib.neo4j.hybrid_context.select_rewrite_model",
        return_value="gemini-2.5-flash",
    ):
        rewritten = await rewrite_search_query_with_occasion(
            "cake for mom",
            "Birthday",
            genai_client=mock_client,
        )

    assert rewritten == "birthday cake for mom"
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert "cake for mom" in call_kwargs["contents"]
    assert "Birthday" in call_kwargs["contents"]


@pytest.mark.asyncio
async def test_rewrite_search_query_uses_lora_endpoint_when_configured() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = RewrittenSearchQuery(q="avurudu cake gifts")
    mock_client.models.generate_content.return_value = mock_response
    lora_model = "projects/test/locations/us-central1/endpoints/lora-rewrite"

    with patch(
        "lib.neo4j.hybrid_context.select_rewrite_model",
        return_value=lora_model,
    ):
        rewritten = await rewrite_search_query_with_occasion(
            "cake ona",
            "Avurudu",
            genai_client=mock_client,
        )

    assert rewritten == "avurudu cake gifts"
    assert mock_client.models.generate_content.call_args.kwargs["model"] == lora_model


@pytest.mark.asyncio
async def test_rewrite_search_query_skips_when_occasion_already_in_query() -> None:
    mock_client = MagicMock()

    rewritten = await rewrite_search_query_with_occasion(
        "birthday cake for mom",
        "Birthday",
        genai_client=mock_client,
    )

    assert rewritten == "birthday cake for mom"
    mock_client.models.generate_content.assert_not_called()
