"""Kapruka product ID detection — regex fast-path without LLM."""

from __future__ import annotations

import re

# Kapruka product IDs often embed digits (e.g. cake00ka002034, EF_PC_CHOC0V2774P00065).
PRODUCT_ID_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*\d[A-Za-z0-9_]{2,})\b")


def contains_product_id(message: str) -> bool:
    """Return True when message contains a Kapruka-like product id token."""
    return PRODUCT_ID_RE.search(message) is not None


def extract_product_id(message: str) -> str | None:
    """Return the first Kapruka-like product id token in the message, if any."""
    match = PRODUCT_ID_RE.search(message)
    return match.group(1) if match else None


def is_valid_product_id(product_id: str) -> bool:
    """Return True when product_id matches Kapruka catalog id format."""
    return bool(PRODUCT_ID_RE.fullmatch(product_id.strip()))
