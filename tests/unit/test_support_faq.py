"""Unit tests for support FAQ detection and copy."""

from __future__ import annotations

from lib.chat.support_faq import (
    build_support_faq_reply,
    classify_support_topic,
    is_support_question,
)
from lib.chat.system_prompts import build_general_welcome_message


def test_is_support_question_matches_return_policy_wilted_flowers() -> None:
    message = "What's your return policy if flowers arrive wilted?"
    assert is_support_question(message)
    assert classify_support_topic(message) == "quality"


def test_is_support_question_matches_refund_policy() -> None:
    assert is_support_question("What is your refund policy for damaged cakes?")


def test_is_support_question_does_not_match_product_search() -> None:
    assert not is_support_question("fresh roses for mom in Galle")


def test_build_support_faq_reply_includes_handoff_not_welcome() -> None:
    message = "What's your return policy if flowers arrive wilted?"
    reply = build_support_faq_reply(message)
    welcome = build_general_welcome_message()
    assert "Kapruka support" in reply
    assert "+94-11-7551111" in reply
    assert "kapruka.com" in reply
    assert reply != welcome
    assert "Welcome to Kapruka" not in reply
