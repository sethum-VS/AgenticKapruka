"""Unit tests for budget refinement and topic pivot heuristics."""

from __future__ import annotations

from lib.chat.intent_heuristics import (
    has_explicit_budget_constraint,
    is_bare_category_pivot,
    is_budget_refinement_message,
    is_natural_budget_gift_message,
    is_topic_pivot_message,
)


def test_is_budget_refinement_message_under_price_only() -> None:
    assert is_budget_refinement_message("under 6000")
    assert is_budget_refinement_message("under Rs. 5,000")
    assert is_budget_refinement_message("Keep it under 6000 rupees.")
    assert is_budget_refinement_message("Keep it under 6000 rupees.")


def test_is_budget_refinement_message_rejects_new_product() -> None:
    assert not is_budget_refinement_message("chocolate gifts under 6000")


def test_is_topic_pivot_message_nevermind_cakes() -> None:
    assert is_topic_pivot_message("Nevermind. Cakes.")
    assert is_topic_pivot_message("cakes")


def test_is_topic_pivot_message_rejects_full_request() -> None:
    assert not is_topic_pivot_message("birthday cake for mom under 5000")


def test_is_bare_category_pivot_nevermind_cakes() -> None:
    assert is_bare_category_pivot("Nevermind. Cakes.") == "cake"
    assert is_bare_category_pivot("cakes") == "cake"
    assert is_bare_category_pivot("birthday cake for mom") is None


def test_has_explicit_budget_constraint_anniversary_under_6000() -> None:
    assert has_explicit_budget_constraint("anniversary gifts under 6000")


def test_has_explicit_budget_constraint_chocolate_under_6000() -> None:
    assert has_explicit_budget_constraint("chocolate for wife under 6000")


def test_has_explicit_budget_constraint_false_on_topic_pivot() -> None:
    assert not has_explicit_budget_constraint(
        "Nevermind. Cakes.",
        6000.0,
        topic_pivot=True,
    )


def test_is_natural_budget_gift_message_wife_budget() -> None:
    assert is_natural_budget_gift_message("wife, budget around 5000 rupees")
    assert not is_natural_budget_gift_message("birthday cake for mom under 5000")
    assert is_natural_budget_gift_message("Gift ideas under Rs. 5,000")
