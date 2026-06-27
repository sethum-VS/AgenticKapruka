"""Unit tests for graphs.nodes.retrieve_hybrid_context."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.call_mcp_tools import select_tool_calls
from graphs.nodes.retrieve_hybrid_context import (
    _fetch_graph_hybrid_context,
    retrieve_hybrid_context,
    route_after_analyze_intent,
)
from graphs.state import AgentState, Intent
from lib.neo4j.client import Neo4jClient
from lib.neo4j.hybrid_context import VECTOR_CONFIDENCE_THRESHOLD
from lib.neo4j.traverse import TraversalResult
from lib.neo4j.vector_search import VectorSearchHit


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_returns_empty_without_graph_or_zep() -> None:
    """No Neo4j client or Zep facts yields empty hybrid_context for generic turns."""
    state: AgentState = {
        "messages": [HumanMessage(content="show me gifts")],
        "intent": "discovery",
        "session_id": "sess-hybrid-001",
    }

    result = await retrieve_hybrid_context(state)

    assert result == {"hybrid_context": {}}


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_birthday_cake_hints_without_graph() -> None:
    """Birthday cake turns still get dessert demotion hints when Neo4j is unavailable."""
    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "intent": "discovery",
        "session_id": "sess-hybrid-birthday",
    }

    result = await retrieve_hybrid_context(state)

    assert "Chocolate" in result["hybrid_context"]["hints"]["exclude_categories"]


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_merges_preferences_from_memory_facts() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="show me gifts")],
        "intent": "discovery",
        "session_id": "sess-hybrid-002",
        "zep_memory_facts": ["User prefers Birthday cakes"],
    }

    result = await retrieve_hybrid_context(state)

    assert result["hybrid_context"]["preferences"]["favorite_category"] == "Birthday"
    assert result["hybrid_context"]["hints"]["category"] == "Birthday"


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_injects_currency_from_preferences() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="show me gifts")],
        "intent": "discovery",
        "session_id": "sess-hybrid-003",
        "zep_memory_facts": ["User shops in USD"],
    }

    result = await retrieve_hybrid_context(state)

    assert result["currency"] == "USD"
    assert result["hybrid_context"]["hints"]["currency"] == "USD"


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_calls_extract_preferences_with_zep_client() -> None:
    zep_client = AsyncMock()
    with patch(
        "graphs.nodes.retrieve_hybrid_context.extract_preferences",
        new=AsyncMock(return_value={"favorite_category": "Flowers"}),
    ) as mock_extract:
        state: AgentState = {
            "messages": [],
            "intent": "discovery",
            "zep_thread_id": "thread-hybrid-004",
        }

        result = await retrieve_hybrid_context(state, zep_client=zep_client)

    mock_extract.assert_awaited_once_with(zep_client, "thread-hybrid-004")
    assert result["hybrid_context"]["hints"]["category"] == "Flowers"


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_merges_graph_and_zep_hints() -> None:
    neo4j_client = AsyncMock(spec=Neo4jClient)
    with patch(
        "graphs.nodes.retrieve_hybrid_context._fetch_graph_hybrid_context",
        new=AsyncMock(
            return_value={
                "hints": {"category": "Flowers", "occasion": "Wedding"},
                "vector_hits": [
                    {"id": "category:flowers", "score": 0.9, "display_name": "Flowers"},
                ],
            }
        ),
    ):
        state: AgentState = {
            "messages": [HumanMessage(content="wedding flowers")],
            "intent": "discovery",
            "session_id": "sess-hybrid-005",
            "zep_memory_facts": ["User shops in USD"],
        }

        result = await retrieve_hybrid_context(state, neo4j_client=neo4j_client)

    hints = result["hybrid_context"]["hints"]
    assert hints["category"] == "Flowers"
    assert hints["occasion"] == "Wedding"
    assert hints["currency"] == "USD"
    assert result["currency"] == "USD"


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_zep_category_overrides_graph_hint() -> None:
    neo4j_client = AsyncMock(spec=Neo4jClient)
    with patch(
        "graphs.nodes.retrieve_hybrid_context._fetch_graph_hybrid_context",
        new=AsyncMock(return_value={"hints": {"category": "Flowers"}}),
    ):
        state: AgentState = {
            "messages": [HumanMessage(content="wedding flowers")],
            "intent": "discovery",
            "session_id": "sess-hybrid-006",
            "zep_memory_facts": ["User prefers Birthday cakes"],
        }

        result = await retrieve_hybrid_context(state, neo4j_client=neo4j_client)

    assert result["hybrid_context"]["hints"]["category"] == "Birthday"


@pytest.mark.parametrize(
    ("intent", "expected_route"),
    [
        ("discovery", "retrieve_hybrid_context"),
        ("general", "retrieve_hybrid_context"),
        ("tracking", "call_mcp_tools"),
        ("checkout", "run_checkout_graph"),
        ("cart", "resolve_cart_product"),
    ],
)
def test_route_after_analyze_intent_routes_checkout_and_skips_tracking(
    intent: Intent,
    expected_route: str,
) -> None:
    state: AgentState = {
        "messages": [],
        "intent": intent,
        "session_id": "sess-route-001",
    }
    assert route_after_analyze_intent(state) == expected_route


def test_route_after_analyze_intent_defaults_to_retrieve_when_intent_missing() -> None:
    state: AgentState = {"messages": [], "session_id": "sess-route-002"}
    assert route_after_analyze_intent(state) == "retrieve_hybrid_context"


def test_route_after_analyze_intent_duplicate_proceed_skips_checkout_graph() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="Proceed to checkout")],
        "intent": "general",
        "intent_metadata": {"duplicate_checkout_proceed": True},
        "session_id": "sess-route-dup-proceed",
    }
    assert route_after_analyze_intent(state) == "generate_response"


def test_route_after_analyze_intent_support_topic_skips_hybrid_context() -> None:
    state: AgentState = {
        "messages": [
            HumanMessage(content="What's your return policy if flowers arrive wilted?"),
        ],
        "intent": "general",
        "intent_metadata": {"support_topic": "quality"},
        "session_id": "sess-route-support",
    }
    assert route_after_analyze_intent(state) == "generate_response"


def test_route_after_analyze_intent_support_with_delivery_routes_preflight() -> None:
    state: AgentState = {
        "messages": [
            HumanMessage(
                content="What's the delivery fee to Colombo and what's your return policy?",
            ),
        ],
        "intent": "general",
        "intent_metadata": {
            "support_topic": "returns",
            "requires_delivery_validation": True,
            "target_city": "Colombo 03",
        },
        "session_id": "sess-route-support-delivery",
    }
    assert route_after_analyze_intent(state) == "resolve_delivery_context"


def test_route_after_analyze_intent_delivery_only_routes_preflight() -> None:
    state: AgentState = {
        "messages": [
            HumanMessage(
                content=(
                    "Can you deliver to Colombo 05 this Sunday? What's the delivery fee?"
                ),
            ),
        ],
        "intent": "discovery",
        "specificity_band": "clarify",
        "agent_clarifying_question": "What type of gift — flowers, cake, voucher, or hamper?",
        "intent_metadata": {
            "requires_delivery_validation": True,
            "target_city": "Colombo 05",
        },
        "session_id": "sess-route-delivery-only",
    }
    assert route_after_analyze_intent(state) == "resolve_delivery_context"


def test_route_after_analyze_intent_specificity_clarify_skips_hybrid_context() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="I want to buy something nice")],
        "intent": "discovery",
        "specificity_band": "clarify",
        "agent_clarifying_question": (
            "What type of gift — flowers, cake, voucher, or hamper?"
        ),
        "session_id": "sess-route-specificity",
    }
    assert route_after_analyze_intent(state) == "generate_response"


@pytest.mark.parametrize("intent", ["discovery", "general"])
def test_route_after_analyze_intent_product_id_skips_hybrid_context(intent: Intent) -> None:
    """Product ID in message bypasses retrieve_hybrid_context (and future agent_loop)."""
    state: AgentState = {
        "messages": [HumanMessage(content="tell me about cake00ka002034")],
        "intent": intent,
        "session_id": "sess-route-product-id",
    }
    assert route_after_analyze_intent(state) == "call_mcp_tools"


@pytest.mark.asyncio
async def test_fetch_graph_hybrid_context_runs_parallel_vector_searches() -> None:
    """Category and Occasion indexes are queried concurrently with one embedding."""
    neo4j_client = AsyncMock(spec=Neo4jClient)
    category_hits = [VectorSearchHit(id="category:flowers", score=0.88)]
    occasion_hits = [
        VectorSearchHit(id="occasion:wedding", score=0.91),
        VectorSearchHit(id="occasion:birthday", score=0.4),
    ]
    embed_fn = AsyncMock(return_value=[[0.1] * 768])

    with (
        patch(
            "graphs.nodes.retrieve_hybrid_context.vector_search",
            new=AsyncMock(return_value=category_hits),
        ) as mock_category_search,
        patch(
            "graphs.nodes.retrieve_hybrid_context.occasion_vector_search",
            new=AsyncMock(return_value=occasion_hits),
        ) as mock_occasion_search,
        patch(
            "graphs.nodes.retrieve_hybrid_context.fetch_category_ids_for_occasions",
            new=AsyncMock(return_value=["category:cakes"]),
        ) as mock_occasion_hop,
        patch(
            "graphs.nodes.retrieve_hybrid_context.fetch_category_display_names",
            new=AsyncMock(
                return_value={
                    "category:flowers": "Flowers",
                    "category:cakes": "Cakes",
                }
            ),
        ),
        patch(
            "graphs.nodes.retrieve_hybrid_context.traverse_from_categories",
            new=AsyncMock(return_value=TraversalResult(nodes=())),
        ) as mock_traverse,
        patch(
            "graphs.nodes.retrieve_hybrid_context.build_graph_hybrid_context",
            return_value={"hints": {"category": "Flowers"}},
        ) as mock_build,
        patch(
            "graphs.nodes.retrieve_hybrid_context.get_reranker",
        ) as mock_get_reranker,
        patch(
            "graphs.nodes.retrieve_hybrid_context.get_settings",
        ) as mock_get_settings,
    ):
        mock_get_settings.return_value.reranker_threshold = 0.45
        result = await _fetch_graph_hybrid_context(
            "wedding flowers",
            neo4j_client=neo4j_client,
            embed_fn=embed_fn,
        )

    embed_fn.assert_awaited_once_with(["wedding flowers"])
    mock_category_search.assert_awaited_once()
    mock_occasion_search.assert_awaited_once()
    assert mock_category_search.await_args.kwargs["top_k"] == 5
    assert mock_occasion_search.await_args.kwargs["top_k"] == 5
    mock_occasion_hop.assert_awaited_once_with(
        neo4j_client,
        ["occasion:wedding"],
    )
    mock_traverse.assert_awaited_once_with(
        neo4j_client,
        ["category:flowers", "category:cakes"],
        max_hops=2,
    )
    mock_get_reranker.assert_called_once()
    mock_build.assert_called_once()
    build_kwargs = mock_build.call_args.kwargs
    assert build_kwargs["vector_hits"] == category_hits
    assert build_kwargs["direct_occasion_hits"] == occasion_hits
    assert build_kwargs["reranker"] is mock_get_reranker.return_value
    assert build_kwargs["reranker_threshold"] == 0.45
    assert result == {"hints": {"category": "Flowers"}}


@pytest.mark.asyncio
async def test_fetch_graph_hybrid_context_skips_low_confidence_occasion_hop() -> None:
    neo4j_client = AsyncMock(spec=Neo4jClient)
    occasion_hits = [VectorSearchHit(id="occasion:birthday", score=0.4)]
    embed_fn = AsyncMock(return_value=[[0.1] * 768])

    with (
        patch(
            "graphs.nodes.retrieve_hybrid_context.vector_search",
            new=AsyncMock(return_value=[VectorSearchHit(id="category:cakes", score=0.7)]),
        ),
        patch(
            "graphs.nodes.retrieve_hybrid_context.occasion_vector_search",
            new=AsyncMock(return_value=occasion_hits),
        ),
        patch(
            "graphs.nodes.retrieve_hybrid_context.fetch_category_ids_for_occasions",
            new=AsyncMock(return_value=[]),
        ) as mock_occasion_hop,
        patch(
            "graphs.nodes.retrieve_hybrid_context.fetch_category_display_names",
            new=AsyncMock(return_value={"category:cakes": "Cakes"}),
        ),
        patch(
            "graphs.nodes.retrieve_hybrid_context.traverse_from_categories",
            new=AsyncMock(return_value=TraversalResult(nodes=())),
        ) as mock_traverse,
        patch(
            "graphs.nodes.retrieve_hybrid_context.build_graph_hybrid_context",
            return_value={"direct_occasion_hits": [{"id": "occasion:birthday", "score": 0.4}]},
        ),
    ):
        await _fetch_graph_hybrid_context(
            "cake",
            neo4j_client=neo4j_client,
            embed_fn=embed_fn,
        )

    mock_occasion_hop.assert_awaited_once_with(neo4j_client, [])
    mock_traverse.assert_awaited_once_with(
        neo4j_client,
        ["category:cakes"],
        max_hops=2,
    )
    assert VECTOR_CONFIDENCE_THRESHOLD == 0.65


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_pruned_graph_hints_only_for_planner() -> None:
    """Hybrid context merges into state as soft hints — discovery MCP args are not auto-built."""
    pruned_graph_context = {
        "hints": {"category": "Flowers"},
        "vector_hits": [
            {"id": "category:flowers", "score": 0.7, "display_name": "Flowers"},
        ],
        "direct_occasion_hits": [],
        "occasions": [],
        "categories": [
            {
                "id": "category:flowers",
                "display_name": "Flowers",
                "hop": 0,
                "relationship_type": "SEED",
                "weight": 1.0,
                "seed_id": "category:flowers",
            },
        ],
        "product_types": [],
    }
    neo4j_client = AsyncMock(spec=Neo4jClient)
    base_state: AgentState = {
        "messages": [HumanMessage(content="something elegant")],
        "intent": "discovery",
    }

    with patch(
        "graphs.nodes.retrieve_hybrid_context._fetch_graph_hybrid_context",
        new=AsyncMock(return_value=pruned_graph_context),
    ):
        updates = await retrieve_hybrid_context(base_state, neo4j_client=neo4j_client)

    mcp_state: AgentState = {**base_state, **updates}
    selected = select_tool_calls(mcp_state)

    assert updates["hybrid_context"]["hints"]["category"] == "Flowers"
    assert selected == []


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_adds_puja_exclude_hints_for_flower_fruit() -> None:
    graph_context = {
        "hints": {"category": "Flowers"},
        "vector_hits": [{"id": "category:flowers", "score": 0.7, "display_name": "Flowers"}],
        "categories": [],
        "occasions": [],
        "product_types": [],
        "direct_occasion_hits": [],
    }
    neo4j_client = AsyncMock(spec=Neo4jClient)
    state: AgentState = {
        "messages": [HumanMessage(content="flowers and fruit basket for Kandy")],
        "intent": "discovery",
    }

    with patch(
        "graphs.nodes.retrieve_hybrid_context._fetch_graph_hybrid_context",
        new=AsyncMock(return_value=graph_context),
    ):
        updates = await retrieve_hybrid_context(state, neo4j_client=neo4j_client)

    assert "exclude_categories" in updates["hybrid_context"]["hints"]
    assert "Puja" in updates["hybrid_context"]["hints"]["exclude_categories"]


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_adds_birthday_dessert_exclude_hints() -> None:
    graph_context = {
        "hints": {"occasion": "Birthday"},
        "vector_hits": [{"id": "category:cakes", "score": 0.7, "display_name": "Cakes"}],
        "categories": [],
        "occasions": [],
        "product_types": [],
        "direct_occasion_hits": [],
    }
    neo4j_client = AsyncMock(spec=Neo4jClient)
    state: AgentState = {
        "messages": [HumanMessage(content="Birthday cake for mom in Colombo")],
        "intent": "discovery",
    }

    with patch(
        "graphs.nodes.retrieve_hybrid_context._fetch_graph_hybrid_context",
        new=AsyncMock(return_value=graph_context),
    ):
        updates = await retrieve_hybrid_context(state, neo4j_client=neo4j_client)

    assert "Chocolate" in updates["hybrid_context"]["hints"]["exclude_categories"]
    assert updates["hybrid_context"]["hints"]["occasion"] == "Birthday"


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_skips_past_occasion_on_topic_pivot() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="Nevermind. Cakes.")],
        "intent": "discovery",
        "intent_metadata": {"topic_pivot": True},
        "zep_memory_facts": ["User celebrated wife's anniversary last year"],
    }

    result = await retrieve_hybrid_context(state)

    hints = result["hybrid_context"].get("hints") or {}
    assert "occasion" not in hints or hints.get("occasion") != "anniversary"


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_wires_anniversary_hints() -> None:
    """Anniversary query populates exclude_categories from enrich_anniversary_hints."""
    state: AgentState = {
        "messages": [HumanMessage(content="anniversary flowers for my wife")],
        "intent": "discovery",
        "session_id": "sess-anniversary-hints",
    }

    updates = await retrieve_hybrid_context(state)

    hints = updates["hybrid_context"].get("hints") or {}
    exclude = str(hints.get("exclude_categories") or "")
    assert exclude, "Expected anniversary exclude_categories hint"
    assert any(
        kw in exclude.lower()
        for kw in ("greeting", "card", "voucher")
    ), f"Expected greeting/voucher in exclude_categories: {exclude!r}"
