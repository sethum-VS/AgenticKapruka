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
    r"\b(?:is that a cake|cupcakes?|"
    r"what(?:'s| is) (?:that|it)|tell me (?:more )?about|"
    r"is (?:that|it) (?:a )?cake|ingredients?|serving size)\b",
    re.I,
)
_PREFERENCE_SWEETNESS_RE = re.compile(
    r"\b(?:less sweet|not too sweet|low sugar|sugar[- ]?free|diabetic)\b",
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


_WEIGHT_NUMERIC = re.compile(r"^[\d.]+$")


def product_weight(product: dict[str, Any]) -> str | None:
    """Return catalog weight when present on a product or detail payload."""
    attributes = product.get("attributes")
    if isinstance(attributes, dict):
        weight = attributes.get("weight")
        if weight is not None and str(weight).strip():
            return str(weight).strip()
    raw_weight = product.get("weight")
    if raw_weight is not None and str(raw_weight).strip():
        return str(raw_weight).strip()
    return None


def _format_weight_display(weight: str) -> str:
    stripped = weight.strip()
    if _WEIGHT_NUMERIC.match(stripped):
        return f"{stripped} Lbs"
    return stripped


def _product_ids_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_id = str(left.get("id") or "")
    right_id = str(right.get("id") or "")
    return bool(left_id and right_id and left_id.upper() == right_id.upper())


def is_valid_product_detail_payload(payload: Any) -> bool:
    """True when payload looks like a successful kapruka_get_product result."""
    return isinstance(payload, dict) and not payload.get("error") and bool(payload.get("name"))


def normalize_resolved_product(product: dict[str, Any]) -> dict[str, Any]:
    """Slim product detail dict safe to persist across turns."""
    normalized: dict[str, Any] = {}
    for key in ("id", "name", "description", "summary", "price", "in_stock", "stock_level", "url"):
        value = product.get(key)
        if value is not None:
            normalized[key] = value
    attributes = product.get("attributes")
    if isinstance(attributes, dict) and attributes:
        normalized["attributes"] = dict(attributes)
    elif product_weight(product):
        normalized["attributes"] = {"weight": product_weight(product)}
    return normalized


def merge_with_session_resolved(
    product: dict[str, Any],
    session_resolved: dict[str, Any] | None,
) -> dict[str, Any]:
    """Overlay persisted MCP detail onto a carousel product when ids match."""
    if not session_resolved:
        return product
    if product.get("id") and session_resolved.get("id"):
        if not _product_ids_match(product, session_resolved):
            return product
    merged = dict(product)
    for key in ("description", "summary", "price", "in_stock", "stock_level", "url"):
        if not merged.get(key) and session_resolved.get(key):
            merged[key] = session_resolved[key]
    resolved_attrs = session_resolved.get("attributes")
    if isinstance(resolved_attrs, dict):
        base_attrs = dict(merged.get("attributes") or {})
        base_attrs.update(resolved_attrs)
        merged["attributes"] = base_attrs
    return merged


def resolve_product_detail(
    *,
    get_payload: dict[str, Any] | None,
    matched: dict[str, Any] | None,
    session_resolved: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return product facts for a detail reply and optional session persistence."""
    if is_valid_product_detail_payload(get_payload):
        normalized = normalize_resolved_product(get_payload)
        return normalized, normalized
    if matched is not None:
        return merge_with_session_resolved(matched, session_resolved), session_resolved
    if session_resolved:
        return session_resolved, session_resolved
    return None, None


def enrich_tool_results_with_session_product(
    tool_results: dict[str, Any] | None,
    session_resolved: dict[str, Any] | None,
    *,
    product_id: str | None = None,
    get_product_tool: str,
) -> dict[str, Any] | None:
    """Inject persisted product detail into LLM context when this turn has no fresh fetch."""
    if not session_resolved:
        return tool_results
    payload = (tool_results or {}).get(get_product_tool)
    if is_valid_product_detail_payload(payload):
        return tool_results
    resolved_id = str(session_resolved.get("id") or "")
    if product_id and resolved_id and product_id.upper() != resolved_id.upper():
        return tool_results
    enriched = dict(tool_results or {})
    enriched[get_product_tool] = session_resolved
    return enriched


def product_preference_note(user_message: str, product: dict[str, Any]) -> str | None:
    """Honest sweetness guidance when the shopper asks about sugar level."""
    if not _PREFERENCE_SWEETNESS_RE.search(user_message.strip()):
        return None
    catalog_text = f"{product.get('description') or ''} {product.get('summary') or ''}".lower()
    if any(
        token in catalog_text
        for token in ("less sweet", "low sugar", "sugar free", "sugar-free", "diabetic")
    ):
        return "The catalog notes this may suit guests who prefer less sweetness."
    return (
        "Kapruka does not list exact sweetness for this cake. "
        "Ribbon cakes are typically buttercream-based; if low sugar matters, "
        "call Kapruka support at +94-11-7551111 before ordering."
    )


def summarize_product_from_carousel(
    product: dict[str, Any],
    *,
    user_message: str | None = None,
) -> str:
    """Short natural-language summary from a cached search or detail product dict."""
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
    product_id = product.get("id")
    if product_id:
        parts.append(f"ID: {product_id}.")
    weight = product_weight(product)
    if weight:
        parts.append(f"Weight: {_format_weight_display(weight)}.")
    if summary:
        parts.append(summary)
    if price_line:
        parts.append(price_line.strip())
    if user_message:
        preference = product_preference_note(user_message, product)
        if preference:
            parts.append(preference)
    return " ".join(parts).strip()
