"""Unit tests for evals.llm_judge rubrics."""

from __future__ import annotations

from evals.llm_judge import (
    score_constraint_fidelity,
    score_intent_preservation,
    score_local_flavor,
    score_mcp_tool_alignment,
    score_visual_fidelity,
)

_PRODUCT_CARD_HTML = """
<div aria-label="Assistant message">
  <div data-testid="product-carousel">
    <div data-testid="product-card"><img src="https://example.com/cake.jpg" /></div>
  </div>
</div>
"""


def test_mcp_tool_alignment_passes_when_expected_called() -> None:
    score = score_mcp_tool_alignment(
        ["kapruka_search_products", "kapruka_check_delivery"],
        ["kapruka_check_delivery"],
    )
    assert score.verdict == "pass"


def test_mcp_tool_alignment_fails_when_missing() -> None:
    score = score_mcp_tool_alignment([], ["kapruka_search_products"])
    assert score.verdict == "fail"


def test_visual_fidelity_requires_product_card() -> None:
    score = score_visual_fidelity(_PRODUCT_CARD_HTML, require_product_card=True)
    assert score.verdict == "pass"


def test_constraint_fidelity_checks_city() -> None:
    score = score_constraint_fidelity(
        '<div aria-label="Assistant message">Delivery to Kandy is available.</div>',
        must_contain=["Kandy"],
    )
    assert score.verdict == "pass"


def test_local_flavor_skips_utility_queries() -> None:
    score = score_local_flavor("Here are cakes.", query_mode="utility")
    assert score.verdict == "pass"


def test_local_flavor_passes_situational_with_sri_lankan_markers() -> None:
    score = score_local_flavor(
        "Aiyo machan, hodata gentle flowers for this moment.",
        query_mode="situational",
        threshold=0.75,
    )
    assert score.score >= 0.75
    assert score.verdict == "pass"


def test_local_flavor_fails_situational_corporate_tone() -> None:
    score = score_local_flavor(
        "Dear valued customer, we regret to inform you about our catalog.",
        query_mode="situational",
        threshold=0.75,
    )
    assert score.score < 0.75
    assert score.verdict == "fail"


def test_intent_preservation_detects_drift() -> None:
    score = score_intent_preservation(
        "<div>Standard milk chocolate box</div>",
        ["nut-free"],
    )
    assert score.verdict == "fail"
