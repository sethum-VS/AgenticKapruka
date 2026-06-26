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
