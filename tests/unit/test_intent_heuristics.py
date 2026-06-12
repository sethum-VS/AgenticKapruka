"""Unit tests for evals.intent_heuristics."""

from __future__ import annotations

from evals.intent_heuristics import infer_intent_from_message

from lib.chat.intent_heuristics import (
    is_checkout_trigger,
    is_proceed_checkout_message,
    is_tracking_guard,
)


def test_infer_tracking_from_order_number() -> None:
    assert infer_intent_from_message("Track order VIMP34456CB2") == "tracking"


def test_infer_checkout_from_delivery_question() -> None:
    assert infer_intent_from_message("Can you deliver to Kandy tomorrow?") == "checkout"


def test_infer_general_from_categories_question() -> None:
    assert infer_intent_from_message("What kinds of gifts can I buy here?") == "general"


def test_infer_discovery_for_product_search() -> None:
    assert infer_intent_from_message("Birthday cake for mom") == "discovery"


def test_tracking_guard_matches_order_number_and_keywords() -> None:
    assert is_tracking_guard("where is order VIMP34456CB2")
    assert is_tracking_guard("track my order please")


def test_checkout_guard_matches_cart_triggers_not_delivery_questions() -> None:
    assert is_checkout_trigger("checkout my cart")
    assert not is_checkout_trigger("Can you deliver to Kandy tomorrow?")


def test_proceed_checkout_message_is_exact_match() -> None:
    assert is_proceed_checkout_message("Proceed to checkout")
    assert not is_proceed_checkout_message("proceed to checkout")
