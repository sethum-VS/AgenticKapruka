"""Unit tests for graph hybrid_context assembly and occasion hint resolution."""

from __future__ import annotations

from lib.neo4j.hybrid_context import (
    VECTOR_CONFIDENCE_THRESHOLD,
    build_graph_hybrid_context,
)
from lib.neo4j.ontology import LABEL_OCCASION
from lib.neo4j.traverse import TraversalNode, TraversalResult
from lib.neo4j.vector_search import VectorSearchHit


def _occasion_node(
    *,
    occasion_id: str,
    display_name: str,
    weight: float,
) -> TraversalNode:
    return TraversalNode(
        id=occasion_id,
        label=LABEL_OCCASION,
        display_name=display_name,
        hop=1,
        relationship_type="OCCASION_TO_CATEGORY",
        weight=weight,
        seed_id="category:flowers",
    )


def test_best_occasion_hint_prefers_high_confidence_vector_hit() -> None:
    """Direct occasion vector hit at or above threshold wins over traversal overlap."""
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
        VectorSearchHit(id="occasion:wedding", score=VECTOR_CONFIDENCE_THRESHOLD),
        VectorSearchHit(id="occasion:birthday", score=0.4),
    ]

    context = build_graph_hybrid_context(
        "cake for mom",
        vector_hits=[VectorSearchHit(id="category:cakes", score=0.9)],
        direct_occasion_hits=direct_hits,
        display_names={"category:cakes": "Cakes"},
        traversal=traversal,
    )

    assert context["hints"]["occasion"] == "Wedding"


def test_best_occasion_hint_falls_back_to_highest_weight_traversal() -> None:
    """Below-threshold vector hits defer to the heaviest traversed occasion node."""
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
    direct_hits = [VectorSearchHit(id="occasion:wedding", score=0.4)]

    context = build_graph_hybrid_context(
        "something elegant",
        vector_hits=[VectorSearchHit(id="category:flowers", score=0.7)],
        direct_occasion_hits=direct_hits,
        display_names={"category:flowers": "Flowers"},
        traversal=traversal,
    )

    assert context["hints"]["occasion"] == "Birthday"


def test_best_occasion_hint_omitted_when_no_vector_or_traversal_signal() -> None:
    """No hints.occasion when vector scores are low and traversal found no occasions."""
    context = build_graph_hybrid_context(
        "gift ideas",
        vector_hits=[VectorSearchHit(id="category:gifts", score=0.8)],
        direct_occasion_hits=[VectorSearchHit(id="occasion:wedding", score=0.3)],
        display_names={"category:gifts": "Gifts"},
        traversal=TraversalResult(nodes=()),
    )

    assert "occasion" not in context.get("hints", {})
    assert context["hints"]["category"] == "Gifts"
