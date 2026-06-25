"""Text normalization helpers for user-visible strings."""

from __future__ import annotations

import html
import re

_MANGLED_NUMERIC_ENTITY = re.compile(r"[Nn]#(\d+);")
_DOUBLE_ENCODED_ENTITY = re.compile(r"&amp;#(\d+);")


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
