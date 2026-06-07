"""Parse shopper chat messages into checkout sub-graph field updates."""

from __future__ import annotations

import re
from typing import Any

from graphs.checkout_state import CHECKOUT_STEP_ORDER, CheckoutState, next_checkout_step
from graphs.state import CheckoutStep

_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_SL_MOBILE_PHONE = re.compile(r"(?:\+94|0)7[0-9]{8}\b")
_REVIEW_CONFIRM = re.compile(
    r"\b(confirm|place\s+(?:my\s+)?order|looks\s+good|proceed|yes)\b",
    re.IGNORECASE,
)


def _collecting_step(state: CheckoutState) -> CheckoutStep:
    """Step that should accept the latest user message."""
    current = state.get("current_step") or "cart"
    step_valid = state.get("step_valid") or {}
    if step_valid.get(current):
        nxt = next_checkout_step(current)
        return nxt if nxt is not None else current
    return current


def apply_chat_message_to_checkout(state: CheckoutState, message: str) -> CheckoutState:
    """Merge parsed fields from a user message into checkout state."""
    text = message.strip()
    if not text:
        return state

    step = _collecting_step(state)
    updates: dict[str, Any] = {}

    if step == "delivery_city":
        updates["delivery_city"] = text[:100]

    elif step == "delivery_date":
        date_match = _ISO_DATE.search(text)
        if date_match:
            updates["delivery_date"] = date_match.group(1)
        remainder = _ISO_DATE.sub("", text).strip(" ,;")
        if len(remainder) >= 3:
            updates["delivery_address"] = remainder[:250]
        if not state.get("delivery_location_type"):
            updates["delivery_location_type"] = "house"

    elif step == "recipient":
        phone_match = _SL_MOBILE_PHONE.search(text)
        if phone_match:
            updates["recipient_phone"] = phone_match.group(0)
            name_part = text[: phone_match.start()].strip(" ,;:-")
            if name_part:
                updates["recipient_name"] = name_part[:80]
        elif "," in text:
            name_part, phone_part = text.split(",", maxsplit=1)
            updates["recipient_name"] = name_part.strip()[:80]
            updates["recipient_phone"] = phone_part.strip()
        else:
            updates["recipient_name"] = text[:80]

    elif step == "sender":
        lower = text.lower()
        updates["sender_anonymous"] = "anonymous" in lower
        name = re.sub(r"\banonymous\b", "", text, flags=re.IGNORECASE).strip(" ,;:-")
        if name:
            updates["sender_name"] = name[:80]
        elif updates["sender_anonymous"]:
            updates["sender_name"] = "Anonymous"

    elif step == "review" and _REVIEW_CONFIRM.search(text):
        updates["action"] = "advance"
        updates["target_step"] = "review"

    merged = dict(state)
    merged.update(updates)
    return merged  # type: ignore[return-value]


def prepare_checkout_invoke_state(state: CheckoutState) -> CheckoutState:
    """Set action/target for the next sub-graph invocation."""
    step = _collecting_step(state)
    prepared = dict(state)
    prepared["current_step"] = step
    if prepared.get("action") != "advance" or prepared.get("target_step") != "review":
        prepared["action"] = "advance"
        prepared["target_step"] = step
    return prepared  # type: ignore[return-value]


def should_chain_finalize(
    prev_step: CheckoutStep,
    new_step: CheckoutStep,
    result: dict[str, Any],
) -> bool:
    """True when review just advanced to finalize and finalize still needs to run."""
    if result.get("validation_errors"):
        return False
    if prev_step == "review" and new_step == "finalize":
        return not result.get("checkout_url")
    return False


def step_index(step: CheckoutStep) -> int:
    return CHECKOUT_STEP_ORDER.index(step)
