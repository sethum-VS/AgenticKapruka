"""Unit tests for budget refinement and topic pivot heuristics."""

from __future__ import annotations

from lib.chat.intent_heuristics import is_budget_refinement_message, is_topic_pivot_message


def test_is_budget_refinement_message_under_price_only() -> None:
    assert is_budget_refinement_message("under 6000")
    assert is_budget_refinement_message("under Rs. 5,000")


def test_is_budget_refinement_message_rejects_new_product() -> None:
    assert not is_budget_refinement_message("chocolate gifts under 6000")


def test_is_topic_pivot_message_nevermind_cakes() -> None:
    assert is_topic_pivot_message("Nevermind. Cakes.")
    assert is_topic_pivot_message("cakes")


def test_is_topic_pivot_message_rejects_full_request() -> None:
    assert not is_topic_pivot_message("birthday cake for mom under 5000")
