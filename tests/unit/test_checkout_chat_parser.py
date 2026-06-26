"""Unit tests for checkout chat message parsing."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from graphs.checkout_state import initial_checkout_state
from lib.checkout.chat_parser import (
    apply_chat_message_to_checkout,
    parse_checkout_details,
    prepare_checkout_invoke_state,
)

_SESSION_ID = "sess-chat-parser-001"


def test_parse_checkout_details_combined_colombo_date_recipient() -> None:
    with patch("lib.chat.delivery_dates.colombo_today", return_value=date(2026, 6, 12)):
        details = parse_checkout_details(
            "Deliver to Colombo 03 on Saturday — recipient Amara, 0771234567, 12 Flower Road",
        )
    assert details.get("delivery_city") == "Colombo 03"
    assert details.get("delivery_date") == "2026-06-13"
    assert details.get("recipient_phone") == "0771234567"


def test_apply_chat_message_first_checkout_entry_applies_all_parsed_fields() -> None:
    state = initial_checkout_state(
        session_id=_SESSION_ID,
        cart_items=[{"product_id": "x", "quantity": 1}],
    )
    state["current_step"] = "cart"
    state["step_valid"] = {"cart": True}

    with patch("lib.chat.delivery_dates.colombo_today", return_value=date(2026, 6, 12)):
        updated = apply_chat_message_to_checkout(
            state,
            "Colombo 03, tomorrow, Amara 0771234567",
        )

    assert updated.get("delivery_city") == "Colombo 03"
    assert updated.get("delivery_date") == "2026-06-13"
    assert updated.get("recipient_phone") == "0771234567"


def test_apply_chat_message_sets_delivery_city_when_cart_valid() -> None:
    state = initial_checkout_state(
        session_id=_SESSION_ID,
        cart_items=[{"product_id": "x", "quantity": 1}],
    )
    state["current_step"] = "cart"
    state["step_valid"] = {"cart": True}

    updated = apply_chat_message_to_checkout(state, "Colombo 03")

    assert updated["delivery_city"] == "Colombo 03"


def test_apply_chat_message_parses_delivery_date_and_address() -> None:
    state = initial_checkout_state(session_id=_SESSION_ID)
    state["current_step"] = "delivery_date"
    state["delivery_city"] = "Colombo 03"

    updated = apply_chat_message_to_checkout(state, "2026-06-10, 123 Galle Road")

    assert updated["delivery_date"] == "2026-06-10"
    assert updated["delivery_address"] == "123 Galle Road"


def test_prepare_checkout_invoke_targets_collecting_step() -> None:
    state = initial_checkout_state(session_id=_SESSION_ID)
    state["current_step"] = "cart"
    state["step_valid"] = {"cart": True}
    state["delivery_city"] = "Colombo 03"

    prepared = prepare_checkout_invoke_state(state)

    assert prepared["current_step"] == "delivery_city"
    assert prepared["action"] == "advance"
    assert prepared["target_step"] == "delivery_city"


def test_review_confirmation_sets_advance_action() -> None:
    state = initial_checkout_state(session_id=_SESSION_ID)
    state["current_step"] = "review"

    updated = apply_chat_message_to_checkout(state, "Yes, please place my order")

    assert updated.get("action") == "advance"
    assert updated.get("target_step") == "review"


def test_review_confirmation_matches_place_the_order() -> None:
    state = initial_checkout_state(session_id=_SESSION_ID)
    state["current_step"] = "review"

    updated = apply_chat_message_to_checkout(state, "place the order")

    assert updated.get("action") == "advance"
    assert updated.get("target_step") == "review"
