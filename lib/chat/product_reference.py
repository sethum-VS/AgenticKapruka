"""Resolve deictic and ordinal product references against session carousel context."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

_ORDINAL_INDEX: dict[str, int] = {
    "first": 0,
    "1st": 0,
    "one": 0,
    "the first": 0,
    "the first one": 0,
    "second": 1,
    "2nd": 1,
    "two": 1,
    "the second": 1,
    "the second one": 1,
    "third": 2,
    "3rd": 2,
    "three": 2,
    "the third": 2,
    "the third one": 2,
    "fourth": 3,
    "4th": 3,
    "the fourth": 3,
    "the fourth one": 3,
    "fifth": 4,
    "5th": 4,
    "the fifth": 4,
    "the fifth one": 4,
}

_DEICTIC_RE = re.compile(
    r"^(?:that|this|it|this one|that one)$",
    re.I,
)


class ProductReferenceResult(TypedDict, total=False):
    status: Literal["resolved", "clarify"]
    product: dict[str, Any] | None
    clarifying_question: str | None
    candidates: list[dict[str, Any]]


def is_deictic_phrase(phrase: str) -> bool:
    """True for pronouns like that, this, it."""
    return bool(_DEICTIC_RE.match(phrase.strip()))


def is_ordinal_phrase(phrase: str) -> bool:
    """True for ordinals like first, second, 1st."""
    normalized = phrase.strip().lower()
    if not normalized:
        return False
    if normalized in _ORDINAL_INDEX:
        return True
    return bool(re.match(r"^(?:the\s+)?\d+(?:st|nd|rd|th)(?:\s+one)?$", normalized))


def _ordinal_index(phrase: str) -> int | None:
    normalized = phrase.strip().lower()
    if normalized in _ORDINAL_INDEX:
        return _ORDINAL_INDEX[normalized]
    match = re.match(r"^(?:the\s+)?(\d+)(?:st|nd|rd|th)(?:\s+one)?$", normalized)
    if match:
        return max(0, int(match.group(1)) - 1)
    return None


def _product_name(product: dict[str, Any]) -> str:
    name = product.get("name")
    return str(name) if name is not None else "item"


def _numbered_clarify(products: list[dict[str, Any]], *, max_items: int = 5) -> str:
    names = [_product_name(product) for product in products[:max_items]]
    numbered = ", ".join(f"{index}) {name}" for index, name in enumerate(names, start=1))
    return f"Which one would you like me to add — {numbered}?"


def _candidate_products(
    *,
    last_visible_products: list[dict[str, Any]] | None,
    last_search_products: list[dict[str, Any]] | None,
    session_product_focus: str | None,
) -> list[dict[str, Any]]:
    visible = [item for item in (last_visible_products or []) if isinstance(item, dict)]
    if visible:
        return visible
    search = [item for item in (last_search_products or []) if isinstance(item, dict)]
    if search:
        return search
    if session_product_focus:
        return []
    return []


def resolve_product_reference(
    phrase: str,
    *,
    last_visible_products: list[dict[str, Any]] | None,
    last_search_products: list[dict[str, Any]] | None,
    session_product_focus: str | None = None,
) -> ProductReferenceResult | None:
    """Resolve deictic/ordinal phrases; return None to fall through to name overlap."""
    stripped = phrase.strip()
    if not stripped:
        return None

    products = _candidate_products(
        last_visible_products=last_visible_products,
        last_search_products=last_search_products,
        session_product_focus=session_product_focus,
    )

    if is_ordinal_phrase(stripped):
        index = _ordinal_index(stripped)
        if index is None:
            return None
        if not products:
            return {
                "status": "clarify",
                "product": None,
                "clarifying_question": (
                    "Search for a gift first, then say which one to add — "
                    "for example, 'add the first one to my cart'."
                ),
            }
        if 0 <= index < len(products):
            return {
                "status": "resolved",
                "product": products[index],
            }
        return {
            "status": "clarify",
            "product": None,
            "clarifying_question": (
                f"I only see {len(products)} option(s) from your last search. "
                "Which product should I add?"
            ),
        }

    if not is_deictic_phrase(stripped):
        return None

    if not products:
        return {
            "status": "clarify",
            "product": None,
            "clarifying_question": (
                "Search for a gift first, then say 'add that to my cart'."
            ),
        }
    if len(products) == 1:
        return {
            "status": "resolved",
            "product": products[0],
        }
    return {
        "status": "clarify",
        "product": None,
        "clarifying_question": _numbered_clarify(products),
        "candidates": products[:5],
    }
