"""Keyword heuristics for intent routing when Gemini is unavailable."""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["discovery", "checkout", "tracking", "general", "cart"]

_ORDER_NUMBER = re.compile(r"\bVIMP[0-9A-Z]+\b", re.I)

_CART_ADD_TO_PATTERN = re.compile(
    r"\badd\s+(.+?)\s+to\s+(?:my\s+)?cart\b",
    re.I,
)
_CART_PUT_IN_PATTERN = re.compile(
    r"\bput\s+(.+?)\s+in(?:to)?\s+(?:my\s+)?cart\b",
    re.I,
)

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
    ),
)

PROCEED_CHECKOUT_MESSAGE = "Proceed to checkout"


def is_cart_add_trigger(message: str) -> bool:
    """Return True for add-to-cart or put-in-cart phrasing."""
    text = message.strip()
    if not text:
        return False
    return bool(_CART_ADD_TO_PATTERN.search(text) or _CART_PUT_IN_PATTERN.search(text))


def extract_cart_product_phrase(message: str) -> str | None:
    """Extract the product phrase from an add-to-cart utterance."""
    text = message.strip()
    if not text:
        return None
    for pattern in (_CART_ADD_TO_PATTERN, _CART_PUT_IN_PATTERN):
        match = pattern.search(text)
        if match:
            phrase = match.group(1).strip(" .,!?:;\"'")
            return phrase or None
    return None


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
    """Return True for explicit checkout/cart-view triggers — not add-to-cart phrases."""
    lowered = message.strip().lower()
    if not lowered:
        return False
    if is_cart_add_trigger(message):
        return False
    if is_proceed_checkout_message(message):
        return True
    return any(token in lowered for token in _CHECKOUT_TRIGGER_TOKENS)


def classify_routing_guard(message: str) -> Intent | None:
    """Shared guard ordering: cart_add → proceed_checkout → tracking → checkout view."""
    if is_cart_add_trigger(message):
        return "cart"
    if is_proceed_checkout_message(message):
        return "checkout"
    if is_tracking_guard(message):
        return "tracking"
    if is_checkout_trigger(message):
        return "checkout"
    return None


def infer_intent_from_message(message: str) -> Intent:
    """Map a user utterance to a shopping-graph intent without calling Gemini."""
    guard = classify_routing_guard(message)
    if guard is not None:
        return guard

    lowered = message.strip().lower()
    if not lowered:
        return "general"

    if any(
        token in lowered
        for token in (
            "checkout",
            "deliver",
            "delivery",
            "pay",
            "recipient",
            "sender",
            "place my order",
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
