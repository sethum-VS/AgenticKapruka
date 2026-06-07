"""Order tracking helpers for chat flow."""

from __future__ import annotations

import re
from typing import Any

from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import TrackOrderOutput

# Pre-payment checkout reference (order_ref) — not valid for kapruka_track_order.
_ORDER_REF_RE = re.compile(r"\bORD-\d{8}-\d+\b", re.IGNORECASE)

# Post-payment Kapruka order numbers (e.g. VIMP34456CB2).
_ORDER_NUMBER_RE = re.compile(
    r"\b([A-Z][A-Z0-9]{2,}[0-9][A-Z0-9]*)\b",
    re.IGNORECASE,
)


def extract_order_number(message: str) -> str | None:
    """Extract post-payment Kapruka order_number from a user message."""
    if _ORDER_REF_RE.search(message):
        return None

    for match in _ORDER_NUMBER_RE.finditer(message):
        candidate = match.group(1).upper()
        if len(candidate) >= 4:
            return candidate
    return None


def tracking_output_from_tool_results(
    tool_results: dict[str, Any] | None,
) -> TrackOrderOutput | None:
    """Return typed tracking payload from call_mcp_tools tool_results."""
    if not tool_results:
        return None
    payload = tool_results.get(TRACK_ORDER_TOOL)
    if not isinstance(payload, dict):
        return None
    return TrackOrderOutput.model_validate(payload)
