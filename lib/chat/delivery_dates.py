"""Parse and validate delivery dates for chat agent grounding (Asia/Colombo)."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from lib.chat.query_preprocessor import extract_target_city
from lib.utils.timezone import colombo_today, colombo_today_iso, is_past_colombo_date

_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

_WEEKDAY_NAMES: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_BARE_WEEKDAY = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)


def _resolve_weekday(today: date, target_weekday: int, *, is_next: bool) -> date:
    """Resolve this/next weekday relative to today in Colombo calendar."""
    days_ahead = (target_weekday - today.weekday()) % 7
    if is_next and days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def _saturday_for_weekend(today: date, *, next_week: bool) -> date:
    """Saturday anchor for this/next weekend phrasing."""
    if not next_week:
        if today.weekday() == 6:
            return today
        days_to_saturday = (5 - today.weekday()) % 7
        return today + timedelta(days=days_to_saturday)

    days_to_saturday = (5 - today.weekday()) % 7
    if today.weekday() >= 5:
        days_to_saturday += 7
    else:
        days_to_saturday += 7
    return today + timedelta(days=days_to_saturday)


def parse_relative_delivery_date(text: str, *, today: date | None = None) -> date | None:
    """Parse relative delivery phrases (next Saturday, this weekend) from natural language."""
    if today is None:
        today = colombo_today()

    normalized = text.strip().lower()
    if not normalized:
        return None

    iso_match = _ISO_DATE.search(text)
    if iso_match:
        try:
            return date.fromisoformat(iso_match.group(1))
        except ValueError:
            return None

    if re.search(r"\btoday\b", normalized):
        return today

    if re.search(r"\btomorrow\b", normalized):
        return today + timedelta(days=1)

    if "this weekend" in normalized:
        return _saturday_for_weekend(today, next_week=False)
    if "next weekend" in normalized:
        return _saturday_for_weekend(today, next_week=True)

    for prefix, is_next in (("next", True), ("this", False)):
        for name, weekday in _WEEKDAY_NAMES.items():
            if re.search(rf"\b{prefix}\s+{name}\b", normalized):
                return _resolve_weekday(today, weekday, is_next=is_next)

    bare_match = _BARE_WEEKDAY.search(normalized)
    if bare_match:
        weekday = _WEEKDAY_NAMES[bare_match.group(1).lower()]
        return _resolve_weekday(today, weekday, is_next=True)

    return None


def validate_delivery_date_iso(value: str) -> tuple[bool, str | None]:
    """Return (ok, error_message). ok=True when value is YYYY-MM-DD on or after Colombo today."""
    stripped = value.strip()
    if not _ISO_DATE.fullmatch(stripped):
        return False, "delivery_date must be YYYY-MM-DD"
    try:
        date.fromisoformat(stripped)
    except ValueError:
        return False, "delivery_date must be YYYY-MM-DD"
    if is_past_colombo_date(stripped):
        return False, "delivery_date cannot be in the past"
    return True, None


def _date_from_raw_value(raw: str, *, today: date) -> str | None:
    """Resolve a single raw date string to validated YYYY-MM-DD or None."""
    iso_match = _ISO_DATE.search(raw)
    if iso_match:
        candidate = iso_match.group(1)
        ok, _ = validate_delivery_date_iso(candidate)
        return candidate if ok else None

    parsed = parse_relative_delivery_date(raw, today=today)
    if parsed is None:
        return None
    if parsed < today:
        return None
    return parsed.isoformat()


def normalize_delivery_date(
    tool_args: dict[str, Any],
    user_message: str,
    *,
    today: date | None = None,
) -> str | None:
    """Resolve delivery_date from tool args and/or user message to YYYY-MM-DD or None."""
    if today is None:
        today = colombo_today()

    for key in ("delivery_date", "date"):
        raw = tool_args.get(key)
        if isinstance(raw, str) and raw.strip():
            resolved = _date_from_raw_value(raw, today=today)
            if resolved is not None:
                return resolved

    resolved_from_message = _date_from_raw_value(user_message, today=today)
    if resolved_from_message is not None:
        return resolved_from_message

    parsed = parse_relative_delivery_date(user_message, today=today)
    if parsed is not None and parsed >= today:
        return parsed.isoformat()

    return None


def is_delivery_date_only_message(text: str, *, today: date | None = None) -> bool:
    """True when the message names a delivery date but no destination city."""
    if normalize_delivery_date({}, text, today=today) is None:
        return False
    return extract_target_city(text) is None


def delivery_date_clarifying_question() -> str:
    """User-facing prompt when delivery date is missing or invalid."""
    today_iso = colombo_today_iso()
    return (
        f"When would you like delivery? Please share a date on or after {today_iso} "
        "(for example YYYY-MM-DD or next Saturday)."
    )
