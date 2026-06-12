"""Keyword heuristics for intent routing when Gemini is unavailable."""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["discovery", "checkout", "tracking", "general"]

_ORDER_NUMBER = re.compile(r"\bVIMP[0-9A-Z]+\b", re.I)

_TRACKING_GUARD_TOKENS: frozenset[str] = frozenset(
    ("track", "where is my order", "order status", "shipped"),
)

_CHECKOUT_TRIGGER_TOKENS: frozenset[str] = frozenset(
    (
        "checkout",
        "check out",
        "place my order",
        "place order",
        "my cart",
        "view cart",
        "pay now",
        "complete my order",
        "proceed to payment",
        "recipient details",
        "sender details",
        "cities near",
        "delivery cities",
    ),
)

PROCEED_CHECKOUT_MESSAGE = "Proceed to checkout"


def is_proceed_checkout_message(message: str) -> bool:
    """Return True for the exact cart-drawer proceed-to-checkout trigger."""
    return message.strip() == PROCEED_CHECKOUT_MESSAGE


def is_tracking_guard(message: str) -> bool:
    """Return True when message matches tracking fast-path (order number or track keywords)."""
    lowered = message.strip().lower()
    if not lowered:
        return False
    return bool(_ORDER_NUMBER.search(message)) or any(
        token in lowered for token in _TRACKING_GUARD_TOKENS
    )


def is_checkout_trigger(message: str) -> bool:
    """Return True for explicit checkout/cart triggers — not discovery delivery questions."""
    lowered = message.strip().lower()
    if not lowered:
        return False
    if is_proceed_checkout_message(message):
        return True
    return any(token in lowered for token in _CHECKOUT_TRIGGER_TOKENS)


def infer_intent_from_message(message: str) -> Intent:
    """Map a user utterance to a shopping-graph intent without calling Gemini."""
    lowered = message.strip().lower()
    if not lowered:
        return "general"

    if _ORDER_NUMBER.search(message) or any(
        token in lowered for token in ("track", "where is my order", "order status", "shipped")
    ):
        return "tracking"

    if lowered == "proceed to checkout":
        return "checkout"

    if any(
        token in lowered
        for token in (
            "checkout",
            "deliver",
            "delivery",
            "cart",
            "pay",
            "recipient",
            "sender",
            "place my order",
            "cities near",
            "delivery cities",
        )
    ):
        return "checkout"

    if any(
        token in lowered
        for token in ("categories", "kinds of gifts", "what can i buy", "what do you sell")
    ):
        return "general"

    if lowered.startswith("cake00ka") or "product " in lowered:
        return "discovery"

    return "discovery"
