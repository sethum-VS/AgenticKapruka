"""Unit tests for lib/chat/delivery_dates relative parse and validation."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from lib.chat.delivery_dates import (
    delivery_date_clarifying_question,
    is_delivery_date_only_message,
    normalize_delivery_date,
    parse_relative_delivery_date,
    validate_delivery_date_iso,
)

COLOMBO = ZoneInfo("Asia/Colombo")
_FRIDAY = date(2026, 6, 12)


def test_parse_relative_next_saturday_from_friday() -> None:
    """next Saturday on a Friday resolves to the following day."""
    assert parse_relative_delivery_date("deliver next Saturday to Colombo", today=_FRIDAY) == date(
        2026, 6, 13
    )


def test_parse_relative_this_weekend_from_friday() -> None:
    """this weekend on a Friday resolves to the upcoming Saturday."""
    assert parse_relative_delivery_date("this weekend please", today=_FRIDAY) == date(2026, 6, 13)


def test_parse_relative_next_weekend_from_friday() -> None:
    """next weekend on a Friday resolves to Saturday the week after."""
    assert parse_relative_delivery_date("next weekend delivery", today=_FRIDAY) == date(2026, 6, 20)


def test_parse_relative_tomorrow() -> None:
    assert parse_relative_delivery_date("tomorrow", today=_FRIDAY) == date(2026, 6, 13)


def test_normalize_delivery_date_iso_passthrough() -> None:
    """Valid future ISO in tool args is returned unchanged."""
    future_iso = "2026-06-25"
    resolved = normalize_delivery_date(
        {"city": "Colombo", "delivery_date": future_iso},
        "cakes for mom",
        today=_FRIDAY,
    )
    assert resolved == future_iso


def test_normalize_delivery_date_from_user_message_when_args_past() -> None:
    """Past planner date is ignored when user message has a valid relative date."""
    resolved = normalize_delivery_date(
        {"city": "Colombo", "delivery_date": "2024-06-29"},
        "deliver to Colombo next Saturday",
        today=_FRIDAY,
    )
    assert resolved == "2026-06-13"


def test_normalize_delivery_date_missing_returns_none() -> None:
    """No date in args or message returns None."""
    assert normalize_delivery_date({"city": "Colombo"}, "deliver to Colombo", today=_FRIDAY) is None


def test_validate_delivery_date_iso_rejects_past() -> None:
    fixed = datetime(2026, 6, 12, 12, 0, tzinfo=COLOMBO)
    with patch("lib.utils.timezone.colombo_now", return_value=fixed):
        ok, error = validate_delivery_date_iso("2024-06-29")
    assert ok is False
    assert error == "delivery_date cannot be in the past"


def test_validate_delivery_date_iso_accepts_today() -> None:
    fixed = datetime(2026, 6, 12, 12, 0, tzinfo=COLOMBO)
    with patch("lib.utils.timezone.colombo_now", return_value=fixed):
        ok, error = validate_delivery_date_iso("2026-06-12")
    assert ok is True
    assert error is None


def test_parse_relative_invalid_iso_returns_none() -> None:
    """Malformed ISO tokens do not raise during relative parse."""
    assert parse_relative_delivery_date("deliver 2025-02-30 to Colombo", today=_FRIDAY) is None


def test_delivery_date_clarifying_question_includes_today() -> None:
    fixed = datetime(2026, 6, 12, 12, 0, tzinfo=COLOMBO)
    with patch("lib.utils.timezone.colombo_now", return_value=fixed):
        question = delivery_date_clarifying_question()
    assert "2026-06-12" in question
    assert "next Saturday" in question


def test_is_delivery_date_only_message_true_for_tomorrow() -> None:
    assert is_delivery_date_only_message("tomorrow", today=_FRIDAY) is True


def test_is_delivery_date_only_message_false_when_city_present() -> None:
    assert is_delivery_date_only_message("deliver to Kandy tomorrow", today=_FRIDAY) is False


def test_parse_relative_bare_saturday() -> None:
    """Bare weekday resolves to the next occurrence."""
    friday = date(2026, 6, 12)
    assert parse_relative_delivery_date("Saturday", today=friday) == date(2026, 6, 13)


