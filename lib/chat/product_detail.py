"""Detect product-detail follow-ups and match prior carousel products."""

from __future__ import annotations

import re
from typing import Any

from lib.chat.product_reference import (
    _normalize_ordinal_phrase,
    is_ordinal_phrase,
    resolve_product_reference,
)
from lib.utils.currency import format_currency

_PRODUCT_DETAIL = re.compile(
    r"\b(?:is that a cake|cupcakes?|delivery fee|how much to deliver|"
    r"what(?:'s| is) (?:that|it)|tell me (?:more )?about|"
    r"is (?:that|it) (?:a )?cake|ingredients?|serving size)\b",
    re.I,
)
_DELIVERY_FEE_QUESTION = re.compile(
    r"\b(?:delivery fee|how much to deliver|delivery cost|shipping fee|"
    r"delivery charge|how much.*deliver(?:y)?|cost.*deliver(?:y)?|"
    r"deliver(?:y)? (?:price|rate|amount|charge))\b",
    re.I,
)
_ORDINAL_IN_MESSAGE = re.compile(
    r"\b((?:the\s+)?(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))"
    r"(?:\s+(?:one|\w+))?)\b",
    re.I,
)


def is_product_detail_turn(user_message: str) -> bool:
    """True when the customer asks about a prior product rather than a new search."""
    return bool(user_message.strip() and _PRODUCT_DETAIL.search(user_message))


def is_delivery_fee_question(user_message: str) -> bool:
    """True when the customer asks about delivery cost for a known city/date."""
    return bool(_DELIVERY_FEE_QUESTION.search(user_message))


def _ordinal_phrase_from_message(user_message: str) -> str | None:
    """Extract an ordinal phrase like 'the first cake' from a detail question."""
    match = _ORDINAL_IN_MESSAGE.search(user_message.strip())
    if not match:
        return None
    phrase = _normalize_ordinal_phrase(match.group(1))
    return phrase if is_ordinal_phrase(phrase) else None


def match_product_from_last_search(
    user_message: str,
    last_search_products: list[dict[str, Any]] | None,
    *,
    last_visible_products: list[dict[str, Any]] | None = None,
    session_product_focus: str | None = None,
) -> dict[str, Any] | None:
    """Return the best-matching carousel product for a product-detail follow-up."""
    from graphs.nodes.resolve_cart_product import match_products_by_phrase

    visible = [item for item in (last_visible_products or []) if isinstance(item, dict)]
    search = [item for item in (last_search_products or []) if isinstance(item, dict)]

    ordinal_phrase = _ordinal_phrase_from_message(user_message)
    if ordinal_phrase:
        reference = resolve_product_reference(
            ordinal_phrase,
            last_visible_products=visible or None,
            last_search_products=search or None,
            session_product_focus=session_product_focus,
        )
        if reference is not None and reference.get("status") == "resolved":
            product = reference.get("product")
            if isinstance(product, dict):
                return product

    for candidates in (visible, search):
        if not candidates:
            continue
        product, _tied, _clarify = match_products_by_phrase(user_message, candidates)
        if product is not None:
            return product

    if len(search) == 1:
        return search[0]
    if len(visible) == 1:
        return visible[0]
    return None


def summarize_product_from_carousel(product: dict[str, Any]) -> str:
    """Short natural-language summary from a cached search product dict."""
    name = str(product.get("name") or "that item")
    summary = str(product.get("summary") or product.get("description") or "").strip()
    raw_price = product.get("price")
    price_line = ""
    if isinstance(raw_price, dict):
        amount = raw_price.get("amount")
        currency = raw_price.get("currency") or "LKR"
        if isinstance(amount, (int, float)):
            price_line = f" Price: {format_currency(float(amount), str(currency))}."
    parts = [f"{name}."]
    if summary:
        parts.append(summary)
    if price_line:
        parts.append(price_line.strip())
    return " ".join(parts).strip()
