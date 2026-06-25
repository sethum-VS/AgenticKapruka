"""Parse shopper chat messages into checkout sub-graph field updates."""

from __future__ import annotations

import re
from typing import Any

from graphs.checkout_state import CHECKOUT_STEP_ORDER, CheckoutState, next_checkout_step
from graphs.state import CheckoutStep
from lib.chat.delivery_dates import normalize_delivery_date
from lib.chat.query_preprocessor import extract_target_city

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


_CITY_ONLY = re.compile(
    r"^(?:Colombo(?:\s+\d{2})?|Kandy|Galle|Negombo|Jaffna|Matara)\s*$",
    re.I,
)


def parse_checkout_details(message: str) -> dict[str, Any]:
    """Extract city, date, recipient, and address fields from one checkout message."""
    text = message.strip()
    if not text:
        return {}

    if "," in text and len(text) > 20:
        merged: dict[str, Any] = {}
        for segment in text.split(","):
            piece = parse_checkout_details(segment.strip())
            for key, value in piece.items():
                if value is not None and key not in merged:
                    merged[key] = value
        if merged:
            return merged

    details: dict[str, Any] = {}

    city = extract_target_city(text)
    if not city and _CITY_ONLY.match(text):
        city = text.strip()
    if city:
        details["delivery_city"] = city[:100]

    resolved_date = normalize_delivery_date({}, text)
    if resolved_date:
        details["delivery_date"] = resolved_date

    iso_match = _ISO_DATE.search(text)
    if iso_match and "delivery_date" not in details:
        details["delivery_date"] = iso_match.group(1)

    phone_match = _SL_MOBILE_PHONE.search(text)
    if phone_match:
        details["recipient_phone"] = phone_match.group(0)
        name_part = text[: phone_match.start()].strip(" ,;:-")
        if name_part and len(name_part) >= 2 and not name_part.lower().startswith("deliver"):
            details["recipient_name"] = name_part[:80]

    if "recipient_name" not in details and "," in text:
        name_part, phone_part = text.split(",", maxsplit=1)
        if _SL_MOBILE_PHONE.search(phone_part):
            details["recipient_name"] = name_part.strip()[:80]
            details["recipient_phone"] = _SL_MOBILE_PHONE.search(phone_part).group(0)  # type: ignore[union-attr]

    remainder = _ISO_DATE.sub("", text)
    strip_tokens = (
        city or "",
        str(details.get("recipient_name", "")),
        str(details.get("recipient_phone", "")),
    )
    for token in strip_tokens:
        if isinstance(token, str) and token:
            remainder = remainder.replace(token, "")
    remainder = re.sub(
        r"\b(?:deliver(?:y)?|to|on|for|recipient|phone|address)\b",
        "",
        remainder,
        flags=re.I,
    ).strip(" ,;:-")
    if len(remainder) >= 8 and "delivery_address" not in details and not _CITY_ONLY.match(text):
        details["delivery_address"] = remainder[:250]
        details.setdefault("delivery_location_type", "house")

    return details


def apply_chat_message_to_checkout(state: CheckoutState, message: str) -> CheckoutState:
    """Merge parsed fields from a user message into checkout state."""
    text = message.strip()
    if not text:
        return state

    current = state.get("current_step") or "cart"
    step_valid = state.get("step_valid") or {}
    step = _collecting_step(state)
    updates: dict[str, Any] = {}

    if current == "cart" and step_valid.get("cart"):
        parsed = parse_checkout_details(text)
        for key, value in parsed.items():
            if value is not None and not state.get(key):
                updates[key] = value
        if updates:
            merged = dict(state)
            merged.update(updates)
            return merged  # type: ignore[return-value]

    if step == "delivery_city":
        city = extract_target_city(text) or (_CITY_ONLY.match(text) and text.strip()) or text
        updates["delivery_city"] = str(city)[:100]

    elif step == "delivery_date":
        resolved = normalize_delivery_date({}, text)
        if resolved:
            updates["delivery_date"] = resolved
        else:
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


def step_valid_for_auto_advance(state: CheckoutState) -> bool:
    """True when the current checkout step passed validation and can auto-advance."""
    current = state.get("current_step") or "cart"
    step_valid = state.get("step_valid") or {}
    if current in ("review", "finalize"):
        return False
    return bool(step_valid.get(current))


def should_auto_advance_step(state: CheckoutState) -> bool:
    """Alias for checkout graph auto-advance loop."""
    return step_valid_for_auto_advance(state)


def step_index(step: CheckoutStep) -> int:
    return CHECKOUT_STEP_ORDER.index(step)
