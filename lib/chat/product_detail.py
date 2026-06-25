"""Detect product-detail follow-ups and match prior carousel products."""

from __future__ import annotations

import re
from typing import Any

from graphs.nodes.resolve_cart_product import match_products_by_phrase
from lib.utils.currency import format_currency

_PRODUCT_DETAIL = re.compile(
    r"\b(?:is that a cake|cupcakes?|delivery fee|how much to deliver|"
    r"what(?:'s| is) (?:that|it)|tell me (?:more )?about|"
    r"is (?:that|it) (?:a )?cake|ingredients?|serving size)\b",
    re.I,
)
_DELIVERY_FEE_QUESTION = re.compile(
    r"\b(?:delivery fee|how much to deliver|delivery cost|shipping fee)\b",
    re.I,
)


def is_product_detail_turn(user_message: str) -> bool:
    """True when the customer asks about a prior product rather than a new search."""
    return bool(user_message.strip() and _PRODUCT_DETAIL.search(user_message))


def is_delivery_fee_question(user_message: str) -> bool:
    """True when the customer asks about delivery cost for a known city/date."""
    return bool(_DELIVERY_FEE_QUESTION.search(user_message))


def match_product_from_last_search(
    user_message: str,
    last_search_products: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return the best-matching carousel product for a product-detail follow-up."""
    if not last_search_products:
        return None
    product, _tied, _clarify = match_products_by_phrase(user_message, last_search_products)
    if product is not None:
        return product
    if len(last_search_products) == 1:
        return last_search_products[0]
    return None


def summarize_product_from_carousel(product: dict[str, Any]) -> str:
    """Short natural-language summary from a cached search product dict."""
    name = str(product.get("name") or "that item")
    summary = str(product.get("summary") or "").strip()
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
