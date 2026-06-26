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
    enrich_birthday_cake_hints,
    is_birthday_cake_intent,
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

    assert args["q"] == "birthday cake"
    assert args["max_price"] == 2000.0
    assert args["sort"] == "price_asc"
    assert args["category"] == "Birthday"


def test_build_discovery_search_args_wife_birthday_chocolate_prefers_chocolate_gift() -> None:
    args = build_discovery_search_args(
        "wife birthday chocolate under 6000",
        {"hints": {"occasion": "Birthday"}},
        currency="LKR",
    )

    assert args["q"] == "chocolate birthday cake"
    assert args["max_price"] == 6000.0


def test_is_birthday_cake_intent_detects_explicit_and_occasion_cake() -> None:
    assert is_birthday_cake_intent("Birthday cake for mom in Colombo")
    assert is_birthday_cake_intent("cake for mom's birthday")
    assert not is_birthday_cake_intent("wife birthday chocolate flowers ~8000 LKR")


def test_enrich_birthday_cake_hints_demotes_desserts_for_birthday_cake() -> None:
    context = enrich_birthday_cake_hints(
        "Birthday cake for mom in Colombo",
        {"hints": {"occasion": "Birthday"}},
    )
    assert context["hints"]["occasion"] == "Birthday"
    assert "Chocolate" in context["hints"]["exclude_categories"]
    assert "Desserts" in context["hints"]["exclude_categories"]


def test_enrich_birthday_cake_hints_skips_chocolate_exclusion_when_flavor_requested() -> None:
    context = enrich_birthday_cake_hints(
        "chocolate birthday cake for mom Colombo 8000",
        {"hints": {"occasion": "Birthday"}},
    )
    exclude = context["hints"]["exclude_categories"]
    assert "Chocolate" not in exclude
    assert "Desserts" in exclude


def test_build_discovery_search_args_chocolate_birthday_mom_colombo() -> None:
    args = build_discovery_search_args(
        "chocolate birthday cake for mom in Colombo budget 8000 LKR",
        {"hints": {"occasion": "Birthday"}},
        currency="LKR",
    )
    assert "chocolate" in args["q"].lower()
    assert args["max_price"] == 8000.0
    assert args["sort"] == "price_asc"


def test_enrich_birthday_cake_hints_skips_chocolate_flowers_combo() -> None:
    context = enrich_birthday_cake_hints(
        "wife birthday chocolate flowers ~8000 LKR colombo",
        {"hints": {"occasion": "Birthday"}},
    )
    assert "exclude_categories" not in context.get("hints", {})


def test_build_discovery_search_args_birthday_occasion_meta_browse_prefers_birthday_cake() -> None:
    args = build_discovery_search_args(
        "show me any items",
        {"hints": {"occasion": "Birthday"}},
        currency="LKR",
    )

    assert args["q"] == "birthday cake"
    assert args["category"] == "Birthday"


def test_extract_max_price_parses_tilde_budget() -> None:
    from lib.neo4j.hybrid_context import extract_max_price

    assert extract_max_price("wife birthday chocolate flowers ~8000 LKR colombo") == 8000.0


def test_build_discovery_search_args_strips_trailing_city_from_query() -> None:
    """Location in chat must not pollute Kapruka keyword search."""
    args = build_discovery_search_args(
        "Birthday cake for mom in Colombo",
        {"hints": {"occasion": "Birthday"}},
        currency="LKR",
    )

    assert args["q"] == "Birthday cake for mom"
    assert args["category"] == "Birthday"


def test_canonical_birthday_cake_search_q_prefers_chocolate_variant() -> None:
    from lib.neo4j.hybrid_context import canonical_birthday_cake_search_q

    assert canonical_birthday_cake_search_q("She loves chocolate birthday cake") == (
        "chocolate birthday cake"
    )
    assert canonical_birthday_cake_search_q("birthday cake for mom") == "birthday cake"


def test_build_discovery_search_args_chocolate_birthday_cake_eval_scenario() -> None:
    args = build_discovery_search_args(
        (
            "Hi! I'm looking for a birthday cake for my wife's 30th birthday. "
            "She loves chocolate. Budget around Rs. 3000. We're in Kandy."
        ),
        {"hints": {"occasion": "Birthday"}},
        currency="LKR",
    )
    assert args["q"] == "chocolate birthday cake"
    assert args["category"] == "Birthday"
    assert args["max_price"] == 3000.0


def test_merge_planner_search_args_overrides_planner_catalog_fields() -> None:
    from lib.neo4j.hybrid_context import merge_planner_search_args

    merged = merge_planner_search_args(
        {"q": "chocolate lava cake", "limit": 20, "currency": "LKR"},
        user_message=(
            "Hi! I'm looking for a birthday cake for my wife. She loves chocolate. "
            "Budget around Rs. 3000. We're in Kandy."
        ),
        hybrid_context={"hints": {"occasion": "Birthday"}},
        currency="LKR",
    )
    assert merged["q"] == "chocolate birthday cake"
    assert merged["category"] == "Birthday"
    assert merged["max_price"] == 3000.0
    assert merged["limit"] == 20


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


def test_enrich_flower_fruit_negative_hints_adds_exclude_categories() -> None:
    from lib.neo4j.hybrid_context import enrich_flower_fruit_negative_hints

    context = enrich_flower_fruit_negative_hints(
        "flowers and fruit basket for Kandy",
        {
            "vector_hits": [{"id": "category:flowers", "score": 0.8}],
            "hints": {"category": "Flowers"},
        },
    )
    assert "Puja" in context["hints"]["exclude_categories"]
    assert enrich_flower_fruit_negative_hints("birthday cake", context) == context


def test_enrich_flower_fruit_negative_hints_skips_without_graph_context() -> None:
    from lib.neo4j.hybrid_context import enrich_flower_fruit_negative_hints

    context = enrich_flower_fruit_negative_hints(
        "flowers and fruit basket",
        {"hints": {"category": "Flowers"}},
    )
    assert "exclude_categories" not in context.get("hints", {})


def test_is_broad_cakes_query_matches_bare_cakes_only() -> None:
    from lib.neo4j.hybrid_context import is_broad_cakes_query

    assert is_broad_cakes_query("cakes")
    assert is_broad_cakes_query("Nevermind. Cakes.")
    assert not is_broad_cakes_query("cake for mom")
    assert not is_broad_cakes_query("birthday cake for mom")


def test_build_budget_refinement_search_args_flowers_budget_uses_roses_q() -> None:
    from lib.neo4j.hybrid_context import build_budget_refinement_search_args

    args = build_budget_refinement_search_args(
        {
            "session_product_focus": "flowers",
            "session_budget_max": 5000.0,
            "intent_metadata": {"budget_max": 5000.0},
        },
        "Keep it under 5000 rupees.",
        currency="LKR",
    )
    assert args is not None
    assert args["q"] == "roses"
    assert args["max_price"] == 5000.0


def test_build_budget_refinement_search_args_uses_session_query() -> None:
    from lib.neo4j.hybrid_context import build_budget_refinement_search_args

    args = build_budget_refinement_search_args(
        {
            "session_search_query": "chocolate gift",
            "session_product_focus": "chocolate",
            "session_budget_max": 6000.0,
            "intent_metadata": {"budget_max": 6000.0},
        },
        "under 6000",
        currency="LKR",
    )
    assert args is not None
    assert args["q"] == "chocolate gift"
    assert args["max_price"] == 6000.0


def test_build_budget_refinement_search_args_birthday_chocolate_bias() -> None:
    from lib.neo4j.hybrid_context import build_budget_refinement_search_args

    args = build_budget_refinement_search_args(
        {
            "session_search_query": "chocolate gift",
            "session_product_focus": "chocolate",
            "session_occasion": "birthday",
            "session_budget_max": 6000.0,
            "intent_metadata": {"budget_max": 6000.0},
        },
        "Keep it under 6000 rupees.",
        currency="LKR",
    )
    assert args is not None
    assert args["q"] == "birthday chocolate cake"
    assert args["category"] == "Birthday"
    assert args["sort"] == "relevance"
    assert args["max_price"] == 6000.0


def test_merge_planner_search_args_budget_refinement() -> None:
    from lib.neo4j.hybrid_context import merge_planner_search_args

    merged = merge_planner_search_args(
        {"q": "gift voucher"},
        user_message="under 6000",
        hybrid_context={},
        currency="LKR",
        state={
            "session_search_query": "chocolate gift",
            "session_product_focus": "chocolate",
            "session_budget_max": 6000.0,
            "intent_metadata": {"budget_max": 6000.0},
        },
    )
    assert merged["q"] == "chocolate gift"
    assert merged["max_price"] == 6000.0


def test_build_discovery_search_args_anniversary_bias() -> None:
    from lib.neo4j.hybrid_context import build_discovery_search_args

    args = build_discovery_search_args(
        "Show me some anniversary gifts",
        {"hints": {"occasion": "anniversary"}},
        currency="LKR",
    )
    assert "anniversary" in args["q"].lower()


def test_build_discovery_search_args_topic_pivot_bare_cakes_literal() -> None:
    from lib.neo4j.hybrid_context import build_discovery_search_args

    args = build_discovery_search_args(
        "Nevermind. Cakes.",
        {"hints": {"occasion": "Anniversary"}},
        currency="LKR",
        intent_metadata={"topic_pivot": True},
    )
    assert args["q"] == "cake"
    assert "category" not in args


def test_merge_planner_search_args_topic_pivot_bare_cakes() -> None:
    from lib.neo4j.hybrid_context import merge_planner_search_args

    merged = merge_planner_search_args(
        {"q": "birthday cake", "category": "Birthday"},
        user_message="Nevermind. Cakes.",
        hybrid_context={"hints": {"occasion": "Anniversary"}},
        currency="LKR",
        intent_metadata={"topic_pivot": True},
    )
    assert merged["q"] == "cake"
    assert "category" not in merged


def test_strip_location_from_search_query_for_galle() -> None:
    from lib.neo4j.hybrid_context import strip_location_from_search_query

    assert strip_location_from_search_query("Fresh roses for Galle tomorrow") == (
        "Fresh roses tomorrow"
    )
    assert strip_location_from_search_query("red roses to Galle") == "red roses"


def test_merge_planner_search_args_roses_galle_strips_city() -> None:
    from lib.neo4j.hybrid_context import merge_planner_search_args

    merged = merge_planner_search_args(
        {"q": "roses for Galle", "currency": "LKR"},
        user_message="Fresh roses for Galle tomorrow",
        hybrid_context={},
        currency="LKR",
        intent_metadata={"target_city": "Galle"},
    )
    assert "Galle" not in merged["q"]
    assert "roses" in merged["q"].lower()
