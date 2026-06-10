"""Unit tests for evals.intent_heuristics."""

from __future__ import annotations

from evals.intent_heuristics import infer_intent_from_message


def test_infer_tracking_from_order_number() -> None:
    assert infer_intent_from_message("Track order VIMP34456CB2") == "tracking"


def test_infer_checkout_from_delivery_question() -> None:
    assert infer_intent_from_message("Can you deliver to Kandy tomorrow?") == "checkout"


def test_infer_general_from_categories_question() -> None:
    assert infer_intent_from_message("What kinds of gifts can I buy here?") == "general"


def test_infer_discovery_for_product_search() -> None:
    assert infer_intent_from_message("Birthday cake for mom") == "discovery"
