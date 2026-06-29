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
    enrich_flower_fruit_negative_hints,
    has_strong_hybrid_hints,
    is_birthday_cake_intent,
    is_confident_discovery_turn,
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


# MS MARCO MiniLM outputs raw logits (~-12 to +12), not 0–1 relevance scores.
# Unit tests inject normalized scores; integration tests assert real logit scale.
_MOCK_RERANKER_SCORES = [0.91, 0.44, 0.12]
_RECORDED_MS_MARCO_LOGITS: dict[str, list[float]] = {
    "birthday cake for mom in Colombo": [-9.89, -6.61, -4.04, -1.16],
    "red roses under 5000": [-8.2, -5.1, -3.4],
}


class _SequenceReranker(CrossEncoderService):
    """Return predetermined scores in pair order for unit tests."""

    def __init__(self, scores: list[float]) -> None:
        super().__init__(model=None)  # type: ignore[arg-type]
        self._scores = scores

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        assert len(texts) == len(self._scores)
        return list(self._scores)


def test_reranker_mock_scores_document_normalized_test_scale() -> None:
    """Unit reranker mocks use 0–1 scores; production MS MARCO emits logits (see integration)."""
    assert all(0.0 <= score <= 1.0 for score in _MOCK_RERANKER_SCORES)
    recorded = _RECORDED_MS_MARCO_LOGITS["birthday cake for mom in Colombo"]
    assert all(score < 0 for score in recorded), "MS MARCO logits for ontology text are typically negative"


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
    """No hints when reranker and vector hits are both below confidence."""
    reranker = _SequenceReranker([0.2, 0.1])

    context = build_graph_hybrid_context(
        "gift ideas",
        vector_hits=[VectorSearchHit(id="category:gifts", score=0.5)],
        direct_occasion_hits=[VectorSearchHit(id="occasion:wedding", score=0.3)],
        display_names={"category:gifts": "Gifts"},
        traversal=TraversalResult(nodes=()),
        reranker=reranker,
        reranker_threshold=DEFAULT_RERANKER_THRESHOLD,
    )

    assert context.get("hints") == {}
    assert context["vector_hits"][0]["id"] == "category:gifts"


def test_vector_fallback_hints_when_reranker_scores_low() -> None:
    """High-confidence vector hits seed hints when reranker rejects all candidates."""
    reranker = _SequenceReranker([0.2, 0.1])

    context = build_graph_hybrid_context(
        "gift ideas",
        vector_hits=[VectorSearchHit(id="category:gifts", score=0.82)],
        direct_occasion_hits=[VectorSearchHit(id="occasion:wedding", score=0.3)],
        display_names={"category:gifts": "Gifts"},
        traversal=TraversalResult(nodes=()),
        reranker=reranker,
        reranker_threshold=DEFAULT_RERANKER_THRESHOLD,
    )

    assert context["hints"]["category"] == "Gifts"
    assert "occasion" not in context["hints"]


def test_reranker_relative_ranking_on_ms_marco_logits() -> None:
    """Raw MS MARCO logits still yield hints via relative top-1 ranking."""
    traversal = TraversalResult(
        nodes=(
            _occasion_node(
                occasion_id="occasion:birthday",
                display_name="Birthday",
                weight=1.5,
            ),
        ),
    )
    logits = [-9.89, -6.61, -4.04]
    reranker = _SequenceReranker(logits)

    context = build_graph_hybrid_context(
        "birthday cake for mom in Colombo",
        vector_hits=[
            VectorSearchHit(id="category:mother", score=0.868),
            VectorSearchHit(id="category:cakes", score=0.72),
        ],
        direct_occasion_hits=[],
        display_names={"category:mother": "Mother", "category:cakes": "Cakes"},
        traversal=traversal,
        reranker=reranker,
        reranker_threshold=DEFAULT_RERANKER_THRESHOLD,
    )

    assert context["hints"].get("category") == "Cakes"
    assert context["hints"].get("occasion") == "Birthday"


def test_reranker_strips_city_before_scoring() -> None:
    """Location tokens must not be passed to the cross-encoder."""
    reranker = _SequenceReranker([-4.0, -6.0])
    captured: list[str] = []

    class _CapturingReranker(_SequenceReranker):
        def score_pairs(self, query: str, texts: list[str]) -> list[float]:
            captured.append(query)
            return super().score_pairs(query, texts)

    rerank_and_prune_traversal(
        "birthday cake for mom in Colombo",
        TraversalResult(nodes=()),
        vector_hits=[VectorSearchHit(id="category:cakes", score=0.9)],
        direct_occasion_hits=[VectorSearchHit(id="occasion:birthday", score=0.9)],
        display_names={"category:cakes": "Cakes"},
        reranker=_CapturingReranker([-4.0, -6.0]),
        threshold=DEFAULT_RERANKER_THRESHOLD,
    )

    assert captured == ["birthday cake for mom"]


def test_vendor_occasions_excluded_from_hint_pool() -> None:
    """Hotel/vendor occasion nodes must not enter the rerank candidate pool."""
    traversal = TraversalResult(
        nodes=(
            _occasion_node(
                occasion_id="occasion:amari-colombo",
                display_name="Amari Colombo",
                weight=2.0,
            ),
            _occasion_node(
                occasion_id="occasion:birthday",
                display_name="Birthday",
                weight=1.0,
            ),
        ),
    )
    reranker = _SequenceReranker([-9.89])

    _, _, occasion_hint = rerank_and_prune_traversal(
        "birthday cake for mom",
        traversal,
        vector_hits=[],
        direct_occasion_hits=[],
        display_names={},
        reranker=reranker,
        threshold=DEFAULT_RERANKER_THRESHOLD,
    )

    assert occasion_hint == "Birthday"


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


def test_build_discovery_search_args_tea_gift_under_budget() -> None:
    """Tea gift queries must search for tea, not generic gift/chocolate."""
    args = build_discovery_search_args(
        "tea gift under Rs 5000",
        {},
        currency="LKR",
    )
    assert "tea" in args["q"].lower()
    assert args.get("max_price") == 5000.0
    assert args.get("sort") == "relevance"


def test_build_discovery_search_args_gift_ideas_tea_colleague_uses_relevance_sort() -> None:
    args = build_discovery_search_args(
        "Gift ideas under Rs. 5,000 for a colleague who loves tea",
        {},
        currency="LKR",
    )
    assert "tea" in args["q"].lower()
    assert args.get("max_price") == 5000.0
    assert args.get("sort") == "relevance"


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


def test_build_budget_refinement_skips_when_budgeted_gift_discovery() -> None:
    from lib.neo4j.hybrid_context import build_budget_refinement_search_args

    args = build_budget_refinement_search_args(
        {
            "session_search_query": "birthday cake for mom",
            "session_product_focus": "cake",
            "session_budget_max": 5000.0,
            "intent_metadata": {
                "budget_max": 5000.0,
                "budgeted_gift_discovery": True,
            },
        },
        "wife, budget around 5000 rupees",
        currency="LKR",
    )
    assert args is None


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


def test_build_discovery_search_args_anniversary_flowers_drops_category_filter() -> None:
    from lib.neo4j.hybrid_context import build_discovery_search_args

    args = build_discovery_search_args(
        "what flowers do you have for anniversary?",
        {"hints": {"category": "Flowers", "occasion": "Anniversary"}},
        currency="LKR",
    )

    assert args["q"] == "anniversary flowers"
    assert "category" not in args


def test_merge_planner_search_args_anniversary_flowers_drops_flowers_category() -> None:
    from lib.neo4j.hybrid_context import merge_planner_search_args

    merged = merge_planner_search_args(
        {"q": "anniversary flowers", "category": "Flowers"},
        user_message="what flowers do you have for anniversary?",
        hybrid_context={"hints": {"category": "Flowers", "occasion": "Anniversary"}},
        currency="LKR",
    )

    assert merged["q"] == "anniversary flowers"
    assert "category" not in merged


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


# P0-1 regression: mom-birthday + chocolate + budget must search "chocolate birthday cake"
def test_build_discovery_search_args_mom_birthday_chocolate_budget_uses_chocolate_cake_q() -> None:
    """P0-1: 'birthday gift for mom ... loves chocolate budget 8000' must not return Happy Birthday Mom q."""
    args = build_discovery_search_args(
        "birthday gift for mom Colombo loves chocolate budget 8000",
        {},
        currency="LKR",
    )
    assert "chocolate birthday cake" in args["q"].lower(), f"Expected chocolate birthday cake, got {args['q']!r}"
    assert args.get("category") == "Birthday"
    assert args["max_price"] == 8000.0


def test_build_discovery_search_args_mom_birthday_no_product_falls_back_to_happy_birthday_mom() -> None:
    """P0-1: bare 'birthday gift for mom' with no product focus keeps Happy Birthday Mom."""
    args = build_discovery_search_args(
        "birthday gift for mom Colombo budget 8000",
        {},
        currency="LKR",
    )
    # No chocolate or cake in query — should fall back to generic
    assert args["q"] == "Happy Birthday Mom"
    assert args["max_price"] == 8000.0


def test_build_discovery_search_args_mom_birthday_cake_uses_birthday_cake_q() -> None:
    """P0-1: 'birthday gift for mom, cake' without chocolate → 'birthday cake' q."""
    args = build_discovery_search_args(
        "birthday gift for mom I want a cake budget 5000",
        {},
        currency="LKR",
    )
    assert "birthday cake" in args["q"].lower(), f"Got {args['q']!r}"
    assert args.get("category") == "Birthday"


def test_merge_planner_search_args_birthday_chocolate_gift_overrides_to_cake() -> None:
    """P0-1 regression: planner uses 'chocolate birthday gift' but must be overridden to 'chocolate birthday cake'."""
    from lib.neo4j.hybrid_context import merge_planner_search_args

    merged = merge_planner_search_args(
        {"q": "chocolate birthday gift", "delivery_city": "Colombo"},
        user_message="birthday gift for mom Colombo loves chocolate budget 8000",
        hybrid_context={"hints": {"exclude_categories": "Flower, Flowers, Bouquet, Floral"}, "product_count": 0},
        currency="LKR",
        intent_metadata={"budget_max": 8000.0, "budget_currency": "LKR"},
    )
    assert "cake" in merged["q"].lower(), (
        f"Expected 'chocolate birthday cake' or similar, got {merged['q']!r}"
    )
    assert "condom" not in merged["q"].lower()


def test_birthday_planner_q_needs_override_chocolate_gift_returns_true() -> None:
    """P0-1: planner q 'chocolate birthday gift' needs override for birthday+chocolate query."""
    from lib.neo4j.hybrid_context import _birthday_planner_q_needs_override

    assert _birthday_planner_q_needs_override(
        "chocolate birthday gift",
        "birthday gift for mom Colombo loves chocolate budget 8000",
    )


def test_birthday_planner_q_needs_override_already_correct_returns_false() -> None:
    """P0-1: planner q already set to 'chocolate birthday cake' should NOT be overridden."""
    from lib.neo4j.hybrid_context import _birthday_planner_q_needs_override

    assert not _birthday_planner_q_needs_override(
        "chocolate birthday cake",
        "birthday gift for mom Colombo loves chocolate budget 8000",
    )


def test_merge_planner_search_args_tea_strips_birthday_category() -> None:
    """Tea queries must not inherit a stale Birthday category filter from hybrid hints."""
    from lib.neo4j.hybrid_context import merge_planner_search_args

    merged = merge_planner_search_args(
        {"q": "tea gift", "category": "Birthday", "max_price": 5000},
        user_message="tea gift under Rs 5000",
        hybrid_context={"hints": {"category": "Birthday"}},
        currency="LKR",
        intent_metadata={"budget_max": 5000.0, "budget_currency": "LKR"},
    )
    assert "tea" in merged["q"].lower()
    assert "category" not in merged


def test_has_strong_hybrid_hints_graph_vector_hits() -> None:
    assert not has_strong_hybrid_hints({"vector_hits": [{"id": "category:cakes"}]})
    assert has_strong_hybrid_hints({"hints": {"occasion": "Birthday"}})
    assert not has_strong_hybrid_hints({})


def test_is_confident_discovery_turn_birthday_cake_with_graph() -> None:
    assert is_confident_discovery_turn(
        "birthday cake for mom in Colombo",
        {"vector_hits": [{"id": "category:cakes"}], "hints": {"occasion": "Birthday"}},
        currency="LKR",
    )


def test_is_confident_discovery_turn_rejects_vague_gifts() -> None:
    assert not is_confident_discovery_turn(
        "show me gifts",
        {"hints": {"category": "Birthday"}},
        currency="LKR",
    )


def test_is_confident_discovery_turn_requires_strong_hints() -> None:
    assert is_confident_discovery_turn(
        "red roses under 5000",
        {},
        currency="LKR",
    )
    assert not is_confident_discovery_turn(
        "show me gifts",
        {"hints": {"category": "Birthday"}},
        currency="LKR",
    )
    enriched = enrich_flower_fruit_negative_hints(
        "red roses under 5000",
        {"vector_hits": [{"id": "category:flowers", "score": 0.8}]},
    )
    assert "exclude_categories" in enriched.get("hints", {})
    assert is_confident_discovery_turn(
        "red roses under 5000",
        enriched,
        currency="LKR",
    )
