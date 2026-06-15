"""Product honesty helpers for artificial floral disclosures."""

from __future__ import annotations

import re
from typing import Any

_ARTIFICIAL_FLORAL_TERMS = re.compile(
    r"\b(?:artificial|faux|silk\s+(?:flower|rose|roses|bouquet|floral)|"
    r"(?:soap|paper)\s+(?:flower|rose|roses|bouquet|floral)s?)\b",
    re.I,
)
_SILK_TERM = re.compile(r"\bsilk\b", re.I)
_FLORAL_CONTEXT = re.compile(r"\b(?:flower|rose|roses|bouquet|floral)\b", re.I)
_FLOWERS_REQUEST = re.compile(r"\b(?:flower|flowers|rose|roses|bouquet|floral)s?\b", re.I)

_ARTIFICIAL_PICKS_DISCLAIMER = (
    "Please note: some picks below are silk or artificial floral arrangements, "
    "not fresh-cut flowers."
)


def _product_text(product: dict[str, Any]) -> str:
    parts = [
        str(product.get("name") or ""),
        str(product.get("summary") or ""),
        str(product.get("description") or ""),
    ]
    return " ".join(parts)


def is_artificial_floral(product: dict[str, Any]) -> bool:
    """True when product name/description indicates silk, artificial, soap, or paper florals."""
    text = _product_text(product)
    if _ARTIFICIAL_FLORAL_TERMS.search(text):
        return True
    return bool(_SILK_TERM.search(text) and _FLORAL_CONTEXT.search(text))


def disclaimer_for_product(product: dict[str, Any]) -> str | None:
    """Per-product disclosure when the item is not fresh-cut flowers."""
    if not is_artificial_floral(product):
        return None
    name = str(product.get("name") or "this item").strip()
    return f"'{name}' is a silk or artificial floral arrangement, not fresh-cut flowers."


def is_flowers_request(user_message: str) -> bool:
    """True when the customer turn mentions flowers, roses, or bouquets."""
    return bool(_FLOWERS_REQUEST.search(user_message))


def artificial_floral_note_for_picks(
    products: list[dict[str, Any]],
    *,
    user_message: str = "",
) -> str | None:
    """Proactive disclaimer when top picks include artificial florals on a flowers request."""
    if not is_flowers_request(user_message):
        return None
    picks = products[:3]
    if not any(is_artificial_floral(product) for product in picks):
        return None
    return _ARTIFICIAL_PICKS_DISCLAIMER


def reply_already_discloses_artificial_floral(reply_text: str) -> bool:
    """True when assistant copy already mentions artificial or non-fresh florals."""
    lower = reply_text.lower()
    return any(
        marker in lower
        for marker in (
            "artificial",
            "not fresh",
            "not fresh-cut",
            "silk or artificial",
            "silk/artificial",
        )
    )
