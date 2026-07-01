"""Cross-session preference extraction from Zep memory search."""

from __future__ import annotations

import logging
import re
from typing import Any, Final

from lib.zep.client import ZepClient
from lib.zep.memory import facts_from_context, facts_from_graph_search, get_session_memory_facts

logger = logging.getLogger(__name__)

PREFERENCE_KEYS: Final = frozenset({"favorite_category", "past_occasion", "currency"})

_PREFERENCE_SEARCH_TEXT: Final = "favorite category occasion currency shopping preferences gifts"

_KNOWN_CATEGORIES: Final = (
    "Birthday",
    "Flowers",
    "Cakes",
    "Chocolates",
    "Fruits",
    "Gifts",
    "Anniversary",
    "Valentine",
    "Wedding",
    "Corporate",
)

_CURRENCY_CODES: Final = frozenset({"LKR", "USD", "GBP", "AUD", "CAD", "EUR"})

_CATEGORY_HINT_RE = re.compile(
    r"(?:favorite|prefers?|likes?|shops?\s+for|interested\s+in)\s+"
    r"(?:the\s+)?([A-Za-z][A-Za-z\s']+?)(?:\s+(?:cakes?|flowers?|gifts?|category))?\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(
    r"\b(?:currency|prefers?|shops?\s+in|priced?\s+in)\s+(?:is\s+)?(LKR|USD|GBP|AUD|CAD|EUR)\b",
    re.IGNORECASE,
)
_OCCASION_RE = re.compile(
    r"((?:mom|mother|dad|father|wife|husband|friend)(?:'s)?\s+birthday|"
    r"(?:wedding|anniversary|valentine(?:'s)?\s+day)(?:\s+gift)?)",
    re.IGNORECASE,
)


def _normalize_category(raw: str) -> str | None:
    """Map a free-text category fragment to a known Kapruka category name."""
    cleaned = raw.strip().title()
    for known in _KNOWN_CATEGORIES:
        if known.lower() in cleaned.lower() or cleaned.lower() in known.lower():
            return known
    return cleaned if cleaned else None


def _category_from_fact(fact: str) -> str | None:
    """Extract favorite category from a single Zep fact string."""
    for known in _KNOWN_CATEGORIES:
        if re.search(rf"\b{re.escape(known)}\b", fact, re.IGNORECASE):
            return known

    match = _CATEGORY_HINT_RE.search(fact)
    if match:
        return _normalize_category(match.group(1))
    return None


def _currency_from_fact(fact: str) -> str | None:
    """Extract preferred currency from a single Zep fact string."""
    match = _CURRENCY_RE.search(fact)
    if match:
        code = match.group(1).upper()
        if code in _CURRENCY_CODES:
            return code
    return None


def _occasion_from_fact(fact: str) -> str | None:
    """Extract past occasion context from a single Zep fact string."""
    match = _OCCASION_RE.search(fact)
    if match:
        return match.group(1).strip().lower()
    if re.search(r"\bbirthday\b", fact, re.IGNORECASE):
        return "birthday"
    return None


def parse_preferences_from_facts(facts: list[str]) -> dict[str, str]:
    """Parse structured user preferences from Zep memory fact strings."""
    preferences: dict[str, str] = {}
    generic_occasion: str | None = None

    for fact in facts:
        if not fact:
            continue

        if "favorite_category" not in preferences:
            category = _category_from_fact(fact)
            if category:
                preferences["favorite_category"] = category

        if "currency" not in preferences:
            currency = _currency_from_fact(fact)
            if currency:
                preferences["currency"] = currency

        if "past_occasion" not in preferences:
            occasion = _occasion_from_fact(fact)
            if occasion and occasion != "birthday":
                preferences["past_occasion"] = occasion
            elif occasion == "birthday" and generic_occasion is None:
                generic_occasion = occasion

    if "past_occasion" not in preferences and generic_occasion is not None:
        preferences["past_occasion"] = generic_occasion

    return preferences


async def extract_preferences(
    zep_client: ZepClient,
    zep_thread_id: str,
) -> dict[str, str]:
    """Search Zep memory for user preferences and return structured preference keys."""
    fact_strings: list[str] = []

    try:
        search_response = await zep_client.search_graph(
            query=_PREFERENCE_SEARCH_TEXT,
            user_id=zep_thread_id,
            limit=10,
        )
        fact_strings.extend(facts_from_graph_search(search_response))
    except Exception as exc:
        logger.warning(
            "Zep preference search failed for thread %s: %s",
            zep_thread_id,
            exc,
        )

    if not fact_strings:
        try:
            context_response = await zep_client.get_user_context(zep_thread_id)
            fact_strings = facts_from_context(context_response.context)
        except Exception as exc:
            logger.warning(
                "Zep context fallback failed for thread %s: %s",
                zep_thread_id,
                exc,
            )

    if not fact_strings:
        fact_strings = await get_session_memory_facts(zep_client, zep_thread_id)

    return parse_preferences_from_facts(fact_strings)


def merge_preferences_into_hybrid_context(
    hybrid_context: dict[str, Any] | None,
    preferences: dict[str, str],
    *,
    user_message: str = "",
    topic_pivot: bool = False,
) -> dict[str, Any]:
    """Merge extracted preferences into hybrid_context hints for MCP search."""
    merged: dict[str, Any] = dict(hybrid_context or {})
    if not preferences:
        return merged

    merged["preferences"] = preferences
    hints: dict[str, str] = dict(merged.get("hints") or {})
    if category := preferences.get("favorite_category"):
        hints["category"] = category
    if currency := preferences.get("currency"):
        hints["currency"] = currency
    if occasion := preferences.get("past_occasion"):
        lowered_message = user_message.lower()
        occasion_in_message = occasion.lower() in lowered_message
        if not topic_pivot and occasion_in_message:
            hints["occasion"] = occasion
    merged["hints"] = hints
    return merged
