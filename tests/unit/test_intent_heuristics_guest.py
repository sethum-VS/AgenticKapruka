"""Guest checkout intent heuristics."""

from __future__ import annotations

from lib.chat.intent_heuristics import (
    build_guest_checkout_reply,
    classify_routing_guard,
    is_checkout_trigger,
    is_guest_checkout_question,
)


def test_guest_checkout_question_detected() -> None:
    assert is_guest_checkout_question("Can I checkout as a guest?")
    assert not is_checkout_trigger("Can I checkout as a guest?")


def test_guest_checkout_not_checkout_guard() -> None:
    assert classify_routing_guard("Can I checkout as a guest?") is None


def test_guest_checkout_reply_mentions_click_to_pay() -> None:
    reply = build_guest_checkout_reply(cart_has_items=True)
    assert "guest" in reply.lower()
    assert "click-to-pay" in reply.lower() or "pay" in reply.lower()
