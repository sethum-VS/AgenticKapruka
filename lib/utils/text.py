"""Text normalization helpers for user-visible strings."""

from __future__ import annotations

import html
import re

_MANGLED_NUMERIC_ENTITY = re.compile(r"[Nn]#(\d+);")
_DOUBLE_ENCODED_ENTITY = re.compile(r"&amp;#(\d+);")
_MOJIBAKE_MARKERS = re.compile(r"â€|Ã.|Â.")
_BACKTICK_WRAP = re.compile(r"^`+|`+$")
_MOJIBAKE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u00e2\u20ac\u2122", "'"),
    ("\u00e2\u20ac\u02dc", "'"),
    ("\u00e2\u20ac\u201c", '"'),
    ("\u00e2\u20ac\u009d", '"'),
    ("\u00e2\u20ac\u2013", "–"),
    ("\u00e2\u20ac\u2014", "—"),
)


def decode_html_entities(value: str) -> str:
    """Decode HTML entities from MCP catalog text (e.g. &#8211; → en-dash)."""
    repaired = _DOUBLE_ENCODED_ENTITY.sub(r"&#\1;", value)
    for _ in range(3):
        repaired = _MANGLED_NUMERIC_ENTITY.sub(r"&#\1;", repaired)
        next_value = html.unescape(repaired)
        if next_value == repaired:
            break
        repaired = next_value
    return repaired


def repair_utf8_mojibake(value: str) -> str:
    """Repair common UTF-8 mojibake (latin1 misread as utf-8) when safe."""
    if not value:
        return value
    repaired = value
    for broken, fixed in _MOJIBAKE_REPLACEMENTS:
        repaired = repaired.replace(broken, fixed)
    if not any(ord(char) > 127 for char in repaired):
        return repaired
    try:
        latin_repaired = repaired.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return repaired
    if latin_repaired == repaired:
        return repaired
    if any(ord(char) > 127 for char in latin_repaired) and _MOJIBAKE_MARKERS.search(latin_repaired):
        return repaired
    return latin_repaired


def normalize_catalog_text(value: str) -> str:
    """Mojibake repair plus HTML entity decoding for catalog and reply text."""
    text = decode_html_entities(repair_utf8_mojibake(value))
    text = _BACKTICK_WRAP.sub("", text)
    return re.sub(r"  +", " ", text)


_APOSTROPHE_VARIANTS = re.compile(r"[''`´\u2019\u02bc]|â€™", re.I)
_POSSESSIVE_S_RE = re.compile(r"\b(\w+)s\b", re.I)


def normalize_for_product_match(text: str) -> str:
    """Fold apostrophe/possessive variants for fuzzy catalog name matching."""
    normalized = normalize_catalog_text(text).lower()
    normalized = _APOSTROPHE_VARIANTS.sub("", normalized)
    normalized = _POSSESSIVE_S_RE.sub(r"\1", normalized)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()
