"""Text normalization helpers for user-visible strings."""

from __future__ import annotations

import html


def decode_html_entities(value: str) -> str:
    """Decode HTML entities from MCP catalog text (e.g. &#8211; → en-dash)."""
    decoded = value
    for _ in range(3):
        next_value = html.unescape(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded
