"""Order tracking helpers for chat flow."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import ValidationError

from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import TrackOrderOutput

logger = logging.getLogger(__name__)

OrderReferenceKind = Literal["vimp", "ord_ref", "ka_legacy", "unknown"]

# Pre-payment checkout reference (order_ref) — not valid for kapruka_track_order.
ORD_REF_RE = re.compile(r"\b(ORD-\d{8}-\d+)\b", re.IGNORECASE)

# Post-payment Kapruka order numbers (e.g. VIMP34456CB2).
VIMP_RE = re.compile(r"\b(VIMP[0-9A-Z]+)\b", re.IGNORECASE)

# Legacy Kapruka references (e.g. KA123456, KA-12345678) — may not resolve via MCP.
KA_LEGACY_RE = re.compile(r"\b(KA-?\d{4,})\b", re.IGNORECASE)

# Broader alphanumeric post-payment IDs (non-ORD, non-KA legacy).
_ORDER_NUMBER_RE = re.compile(
    r"\b([A-Z][A-Z0-9]{2,}[0-9][A-Z0-9]*)\b",
    re.IGNORECASE,
)

_VIMP_EXAMPLE = "VIMP34456CB2"


def _normalize_reference(ref: str) -> str:
    """Uppercase and strip hyphens from legacy KA tokens for MCP lookup."""
    return ref.strip().upper().replace("-", "")


def classify_order_reference(ref: str) -> OrderReferenceKind:
    """Classify a single order reference token."""
    text = ref.strip()
    if not text:
        return "unknown"
    if ORD_REF_RE.fullmatch(text):
        return "ord_ref"
    upper = _normalize_reference(text)
    if VIMP_RE.fullmatch(upper) or upper.startswith("VIMP"):
        return "vimp"
    if KA_LEGACY_RE.fullmatch(text) or (upper.startswith("KA") and re.search(r"\d", upper)):
        return "ka_legacy"
    if _ORDER_NUMBER_RE.fullmatch(upper) and len(upper) >= 4:
        return "unknown"
    return "unknown"


def classify_order_references(message: str) -> list[tuple[str, OrderReferenceKind]]:
    """Extract and classify all order-like references in a user message."""
    found: list[tuple[str, OrderReferenceKind]] = []
    seen: set[str] = set()

    for pattern in (ORD_REF_RE, VIMP_RE, KA_LEGACY_RE):
        for match in pattern.finditer(message):
            token = match.group(1)
            key = token.upper()
            if key in seen:
                continue
            seen.add(key)
            found.append((token, classify_order_reference(token)))

    for match in _ORDER_NUMBER_RE.finditer(message):
        token = match.group(1)
        key = token.upper()
        if key in seen or ORD_REF_RE.fullmatch(token):
            continue
        if KA_LEGACY_RE.search(token) or VIMP_RE.search(token):
            continue
        seen.add(key)
        found.append((token, classify_order_reference(token)))

    return found


def extract_order_number(message: str) -> str | None:
    """Extract a trackable Kapruka order_number from a user message."""
    vimp_match = VIMP_RE.search(message)
    if vimp_match:
        return vimp_match.group(1).upper()

    ka_match = KA_LEGACY_RE.search(message)
    if ka_match:
        return _normalize_reference(ka_match.group(1))

    if ORD_REF_RE.search(message) and not vimp_match and not ka_match:
        return None

    for match in _ORDER_NUMBER_RE.finditer(message):
        candidate = match.group(1).upper()
        if len(candidate) >= 4 and not ORD_REF_RE.fullmatch(candidate):
            return candidate
    return None


def build_ord_ref_educate_message() -> str:
    """Explain why pre-payment ORD- references cannot be used for tracking."""
    return (
        "The ORD- reference is from before payment and cannot be used for tracking. "
        f"After you pay, Kapruka emails a post-payment order number (for example "
        f"{_VIMP_EXAMPLE}). Please share that number."
    )


def build_missing_tracking_number_message(user_message: str) -> str:
    """Ask for a post-payment order number, with ORD- guidance when relevant."""
    if ORD_REF_RE.search(user_message):
        return build_ord_ref_educate_message()
    return (
        "Please share your Kapruka order number from your confirmation email. "
        f"Use the post-payment number (for example {_VIMP_EXAMPLE}), not the "
        "pre-payment checkout reference that starts with ORD-."
    )


def build_tracking_failure_message(
    *,
    order_number: str | None,
    reference_kind: OrderReferenceKind,
    error_code: str | None = None,
    raw_message: str | None = None,
) -> str:
    """Tier-1 copy when kapruka_track_order fails — educate on KA legacy, not a hard error."""
    if reference_kind == "ka_legacy" or (
        order_number and classify_order_reference(order_number) == "ka_legacy"
    ):
        legacy = order_number or "that KA reference"
        return (
            f"I could not find order {legacy} in Kapruka's live tracking system. "
            f"Legacy KA-style numbers are not used for tracking after checkout. "
            f"After payment, Kapruka emails a post-payment order number (for example "
            f"{_VIMP_EXAMPLE}) — please share that number from your confirmation email."
        )

    if reference_kind == "ord_ref":
        return build_ord_ref_educate_message()

    if error_code == "order_not_found" or (raw_message and "could not find" in raw_message.lower()):
        if order_number:
            return (
                f"I could not find an order matching {order_number}. "
                f"Please double-check the post-payment order number from your Kapruka "
                f"confirmation email (for example {_VIMP_EXAMPLE})."
            )
        return build_missing_tracking_number_message("")

    cause = (raw_message or "").strip()
    if cause:
        return (
            f"I could not look up that order right now. {cause} "
            f"Please try again with your post-payment order number (for example {_VIMP_EXAMPLE})."
        )
    return build_missing_tracking_number_message("")


def tracking_output_from_tool_results(
    tool_results: dict[str, Any] | None,
) -> TrackOrderOutput | None:
    """Return typed tracking payload from call_mcp_tools tool_results."""
    if not tool_results:
        return None
    payload = tool_results.get(TRACK_ORDER_TOOL)
    if not isinstance(payload, dict):
        return None
    if payload.get("error"):
        return None
    try:
        return TrackOrderOutput.model_validate(payload)
    except ValidationError:
        logger.warning(
            "tracking_output_from_tool_results: invalid %s payload",
            TRACK_ORDER_TOOL,
            exc_info=True,
        )
        return None


def tracking_error_from_tool_results(
    tool_results: dict[str, Any] | None,
) -> dict[str, str] | None:
    """Return MCP error dict from kapruka_track_order tool_results, if present."""
    if not tool_results:
        return None
    payload = tool_results.get(TRACK_ORDER_TOOL)
    if not isinstance(payload, dict) or not payload.get("error"):
        return None
    error_code = payload.get("error")
    message = payload.get("message")
    return {
        "error": str(error_code) if error_code is not None else "unknown",
        "message": str(message).strip() if isinstance(message, str) else "",
    }
