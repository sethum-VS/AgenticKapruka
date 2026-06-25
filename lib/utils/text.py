"""Text normalization helpers for user-visible strings."""

from __future__ import annotations

import html
import re

_MANGLED_NUMERIC_ENTITY = re.compile(r"[Nn]#(\d+);")
_DOUBLE_ENCODED_ENTITY = re.compile(r"&amp;#(\d+);")
_MOJIBAKE_MARKERS = re.compile(r"â€|Ã.|Â.")


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
    if not value or not any(ord(char) > 127 for char in value):
        return value
    try:
        repaired = value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    if repaired == value:
        return value
    if any(ord(char) > 127 for char in repaired) and _MOJIBAKE_MARKERS.search(repaired):
        return value
    return repaired


def normalize_catalog_text(value: str) -> str:
    """Mojibake repair plus HTML entity decoding for catalog and reply text."""
    return decode_html_entities(repair_utf8_mojibake(value))
