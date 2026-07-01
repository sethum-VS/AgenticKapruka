"""Unit tests for lib.chat.request_specificity."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google import genai

from lib.chat.request_specificity import (
    CLARIFY_THRESHOLD,
    PROCEED_THRESHOLD,
    SpecificityRefinement,
    SpecificityResult,
    refine_specificity_with_llm,
    score_request_specificity,
    should_bypass_specificity_scorer,
)


def test_vague_gift_ideas_clarifies_product() -> None:
    result = score_request_specificity(
        "any gift ideas?",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={},
    )
    assert result.band == "clarify"
    assert result.score < CLARIFY_THRESHOLD or result.band == "clarify"
    assert result.missing_dimension == "product"
    assert result.clarifying_question


def test_budgeted_gift_chip_bypasses_scorer() -> None:
    assert should_bypass_specificity_scorer("Gift ideas under Rs. 5,000")


def test_budgeted_gift_chip_proceeds_when_scored() -> None:
    result = score_request_specificity(
        "Gift ideas under Rs. 5,000",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=5000.0,
        intent_metadata={"budget_max": 5000.0},
    )
    assert result.band == "proceed"
    assert result.clarifying_question is None


def test_specific_birthday_cake_proceeds() -> None:
    result = score_request_specificity(
        "birthday cake for mom under Rs 5000",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={},
    )
    assert result.band == "proceed"


def test_gifts_for_mom_proceeds() -> None:
    result = score_request_specificity(
        "gifts for my mom",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={},
    )
    assert result.band == "proceed"


def test_bare_cakes_after_pivot_clarifies_occasion() -> None:
    result = score_request_specificity(
        "cakes",
        session_product_focus="cake",
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={"topic_pivot": True},
    )
    assert result.band == "clarify"
    assert result.missing_dimension == "occasion"


def test_multiturn_chocolate_after_clarify_still_needs_occasion() -> None:
    result = score_request_specificity(
        "chocolate",
        session_product_focus="chocolate",
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={},
    )
    assert result.band == "clarify"
    assert result.missing_dimension == "occasion"


def test_multiturn_chocolate_with_session_occasion_proceeds() -> None:
    result = score_request_specificity(
        "chocolate",
        session_product_focus="chocolate",
        session_occasion="birthday",
        session_recipient_hint="mom",
        session_budget_max=5000.0,
        intent_metadata={},
    )
    assert result.band == "proceed"


def test_empty_message_clarifies() -> None:
    result = score_request_specificity(
        "",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={},
    )
    assert result.band == "clarify"
    assert result.clarifying_question == "What are you looking for today?"


def test_budget_only_without_product_signal_clarifies() -> None:
    result = score_request_specificity(
        "under Rs 5000",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={"budget_max": 5000.0},
    )
    assert result.band in ("clarify", "ambiguous")
    assert result.missing_dimension in ("product", "occasion")
    assert result.clarifying_question


def test_rich_cake_colombo_proceeds_with_zone_clarify() -> None:
    result = score_request_specificity(
        "Mom's 65th birthday this Sunday in Colombo — chocolate cakes under Rs 8,000",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={"target_city": "Colombo", "session_delivery_date": "2026-06-28"},
    )
    assert result.band == "proceed"


def test_roses_galle_proceeds_with_product_and_delivery() -> None:
    result = score_request_specificity(
        "Fresh roses to Galle tomorrow",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={"target_city": "Galle"},
    )
    assert result.band == "proceed"


def test_birthday_cake_mom_colombo_proceeds_without_date() -> None:
    result = score_request_specificity(
        "I need a birthday cake for my mom in Colombo",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={"target_city": "Colombo"},
    )
    assert result.band == "proceed"
    assert result.clarifying_question is None


def test_situational_clarify_prefix() -> None:
    result = score_request_specificity(
        "any gift ideas?",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={"is_situational": True},
    )
    assert result.clarifying_question
    assert result.clarifying_question.startswith("I'm sorry to hear")


@pytest.mark.asyncio
async def test_llm_refine_fallback_to_clarify_on_non_client() -> None:
    heuristic = score_request_specificity(
        "something nice",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={},
    )
    refined = await refine_specificity_with_llm(
        "something nice",
        heuristic,
        genai_client=MagicMock(),
    )
    assert refined.band == "clarify"
    assert refined.clarifying_question


@pytest.mark.asyncio
async def test_llm_refine_proceed_when_model_scores_high() -> None:
    mock_client = MagicMock(spec=genai.Client)
    response = MagicMock()
    response.parsed = SpecificityRefinement(
        score=85.0,
        product_score=1.0,
        occasion_score=0.5,
        budget_score=0.0,
        missing_dimension=None,
        band="proceed",
    )
    mock_client.models.generate_content.return_value = response
    heuristic = SpecificityResult(
        score=35.0,
        dimension_scores={"product": 0.5, "occasion": 0.0, "budget": 0.5},
        missing_dimension="occasion",
        band="ambiguous",
        clarifying_question=None,
    )
    refined = await refine_specificity_with_llm(
        "something nice for a friend",
        heuristic,
        genai_client=mock_client,
    )
    assert refined.band == "proceed"
    assert refined.score >= PROCEED_THRESHOLD


# P1-5 regression: "I want to buy something nice" must NOT inherit stale session context
def test_generic_something_nice_clarifies_despite_stale_session() -> None:
    """P1-5: 'I want to buy something nice' should clarify even with stale birthday/mom session."""
    result = score_request_specificity(
        "I want to buy something nice",
        session_product_focus="chocolate",
        session_occasion="birthday",
        session_recipient_hint="mom",
        session_budget_max=8000.0,
        intent_metadata={"budget_max": 8000.0},
    )
    assert result.band == "clarify", (
        f"Expected clarify for vague 'something nice' but got {result.band!r} "
        f"(score={result.score}, dims={result.dimension_scores})"
    )


def test_something_nice_product_dimension_ignores_stale_focus() -> None:
    """P1-5: 'something nice' product dimension must be 0.0 even with session_product_focus."""
    from lib.chat.request_specificity import _score_product_dimension

    score = _score_product_dimension(
        "I want to buy something nice",
        session_product_focus="chocolate",
        session_flavor_hint=None,
        session_recipient_hint=None,
        intent_metadata={},
    )
    assert score == 0.0, f"Expected 0.0 but got {score}"


def test_something_nice_occasion_dimension_ignores_stale_context() -> None:
    """P1-5: 'something nice' occasion dimension must not inflate from stale session."""
    from lib.chat.request_specificity import _score_occasion_dimension

    score = _score_occasion_dimension(
        "I want to buy something nice",
        session_occasion="birthday",
        session_recipient_hint="mom",
    )
    assert score == 0.0, f"Expected 0.0 but got {score}"


def test_i_need_a_gift_clarifies_despite_stale_session() -> None:
    """P2-11: bare 'I need a gift' must not inherit stale mom/birthday session."""
    result = score_request_specificity(
        "I need a gift",
        session_product_focus="chocolate",
        session_occasion="birthday",
        session_recipient_hint="mom",
        session_budget_max=8000.0,
        intent_metadata={"budget_max": 8000.0},
    )
    assert result.band == "clarify", (
        f"Expected clarify for vague 'I need a gift' but got {result.band!r}"
    )


def test_delivery_only_inquiry_proceeds_without_product_clarify() -> None:
    """Delivery fee + city + date must not trigger product clarifying question."""
    message = "Can you deliver to Colombo 05 this Sunday? What's the delivery fee?"
    metadata = {
        "requires_delivery_validation": True,
        "target_city": "Colombo 05",
    }
    result = score_request_specificity(
        message,
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata=metadata,
    )
    assert result.band == "proceed"
    assert result.clarifying_question is None


def test_category_browse_proceeds_without_clarify() -> None:
    result = score_request_specificity(
        "Hi, what kinds of gifts can I buy here?",
        session_product_focus=None,
        session_occasion=None,
        session_recipient_hint=None,
        session_budget_max=None,
        intent_metadata={},
    )
    assert result.band == "proceed"
    assert result.clarifying_question is None


def test_explicit_product_browse_proceeds_without_occasion() -> None:
    for message in (
        "Show me chocolate options",
        "Maybe flowers would help — something gentle",
        "Cake and flower combo please",
        "Show me chocolate cakes instead",
        "Fresh fruit basket for a get-well gift",
    ):
        result = score_request_specificity(
            message,
            session_product_focus=None,
            session_occasion=None,
            session_recipient_hint=None,
            session_budget_max=None,
            intent_metadata={},
        )
        assert result.band == "proceed", message
