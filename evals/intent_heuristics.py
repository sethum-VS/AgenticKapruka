"""Keyword heuristics for E2E/Ragas mocks when live Gemini is unavailable."""

from __future__ import annotations

import re

from graphs.state import Intent

_ORDER_NUMBER = re.compile(r"\bVIMP[0-9A-Z]+\b", re.I)


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
