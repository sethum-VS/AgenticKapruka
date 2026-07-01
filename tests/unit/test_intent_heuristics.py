"""Unit tests for lib.chat.intent_heuristics cart vs checkout routing."""

from __future__ import annotations

from evals.intent_heuristics import infer_intent_from_message

from lib.chat.intent_heuristics import (
    classify_routing_guard,
    extract_cart_product_phrase,
    is_cart_add_trigger,
    is_checkout_trigger,
    is_order_intent_message,
    is_proceed_checkout_message,
    is_topic_pivot_message,
    is_tracking_guard,
)


def test_cart_add_trigger_matches_add_and_put_phrases() -> None:
    assert is_cart_add_trigger("Add the Blush Roses combo to my cart please")
    assert is_cart_add_trigger("put chocolate cake in my cart")
    assert is_cart_add_trigger("Add roses to cart")


def test_cart_add_trigger_does_not_match_view_cart() -> None:
    assert not is_cart_add_trigger("checkout my cart")
    assert not is_cart_add_trigger("view cart")


def test_extract_cart_product_phrase() -> None:
    assert extract_cart_product_phrase("Add the Blush Roses combo to my cart") == (
        "the Blush Roses combo"
    )
    assert extract_cart_product_phrase("put chocolate cake in my cart") == "chocolate cake"


def test_checkout_trigger_excludes_add_to_cart_substring() -> None:
    assert not is_checkout_trigger("Add Blush Roses to my cart")
    assert is_checkout_trigger("checkout my cart")
    assert is_checkout_trigger("view cart")
    assert is_checkout_trigger("place the order")
    assert is_checkout_trigger("Help me place the order for delivery tomorrow")


def test_is_order_intent_message_matches_place_order_variants() -> None:
    assert is_order_intent_message("place the order")
    assert is_order_intent_message("Please place an order")
    assert is_order_intent_message("Yes, place my order")
    assert not is_order_intent_message("birthday cake for mom")


def test_classify_routing_guard_priority_cart_before_checkout() -> None:
    assert classify_routing_guard("Add Blush Roses to my cart") == "cart"
    assert classify_routing_guard("Proceed to checkout") == "checkout"
    assert classify_routing_guard("where is order VIMP123") == "tracking"
    assert classify_routing_guard("view cart") == "checkout"


def test_infer_intent_add_to_cart_is_cart_not_checkout() -> None:
    assert infer_intent_from_message("Add the Blush Roses combo to my cart please") == "cart"


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
    assert is_tracking_guard("check status of my order")
    assert is_tracking_guard("where is order KA123456")


def test_tracking_guard_matches_ord_ref() -> None:
    assert is_tracking_guard("track ORD-20260520-7823")


def test_infer_general_from_support_return_policy() -> None:
    assert (
        infer_intent_from_message("What's your return policy if flowers arrive wilted?")
        == "general"
    )


def test_proceed_checkout_message_is_exact_match() -> None:
    assert is_proceed_checkout_message("Proceed to checkout")
    assert not is_proceed_checkout_message("proceed to checkout")


def test_what_about_recognized_as_pivot() -> None:
    """'What about cakes?' should be recognized as a topic pivot."""
    assert is_topic_pivot_message("What about cakes?") is True


def test_what_about_flowers_is_pivot() -> None:
    assert is_topic_pivot_message("What about flowers instead?") is True


def test_what_about_with_no_product_is_pivot() -> None:
    assert is_topic_pivot_message("What about something else?") is True
