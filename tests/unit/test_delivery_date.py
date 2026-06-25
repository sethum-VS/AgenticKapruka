"""Structure tests for Colombo timezone delivery date picker."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from app.templating import (
    _create_templates,
    render_delivery_date,
    render_delivery_date_error,
    render_delivery_date_status,
)
from lib.kapruka.types import CheckDeliveryOutput
from lib.utils.timezone import colombo_today, colombo_today_iso, is_past_colombo_date

COLOMBO = ZoneInfo("Asia/Colombo")


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_colombo_today_uses_asia_colombo_zone() -> None:
    """colombo_today_iso matches Asia/Colombo calendar date."""
    fixed = datetime(2026, 6, 8, 1, 30, tzinfo=COLOMBO)
    with patch("lib.utils.timezone.colombo_now", return_value=fixed):
        assert colombo_today() == date(2026, 6, 8)
        assert colombo_today_iso() == "2026-06-08"


def test_is_past_colombo_date_compares_to_colombo_today() -> None:
    """Past dates are detected relative to Colombo today, not UTC."""
    fixed = datetime(2026, 6, 8, 12, 0, tzinfo=COLOMBO)
    with patch("lib.utils.timezone.colombo_now", return_value=fixed):
        assert is_past_colombo_date("2026-06-07") is True
        assert is_past_colombo_date("2026-06-08") is False
        assert is_past_colombo_date("2026-06-09") is False


def test_format_delivery_date_friendly() -> None:
    from lib.utils.timezone import format_delivery_date_friendly

    assert format_delivery_date_friendly("2026-06-28") == "Sunday, 28 June 2026"


def test_delivery_date_input_htmx_check_delivery() -> None:
    """Date input posts to /checkout/check-delivery with Colombo min date."""
    html = render_delivery_date(min_date="2026-06-08")

    assert 'data-testid="delivery-date-field"' in html
    assert 'data-testid="delivery-date-input"' in html
    assert 'name="delivery_date"' in html
    assert 'type="date"' in html
    assert 'min="2026-06-08"' in html
    assert 'hx-post="/checkout/check-delivery"' in html
    assert 'hx-trigger="change"' in html
    assert 'hx-target="#delivery-date-status"' in html
    assert 'hx-swap="innerHTML"' in html
    assert 'hx-include="#delivery-city-value"' in html
    assert 'data-testid="delivery-date-status"' in html
    assert "Asia/Colombo" in html


def test_delivery_date_status_available_partial() -> None:
    """Available check_delivery result renders success partial with rate."""
    result = CheckDeliveryOutput(
        city="Colombo 03",
        now="2026-06-08T10:00:00+05:30",
        checked_date="2026-06-10",
        available=True,
        rate=350.0,
        currency="LKR",
    )
    html = render_delivery_date_status(result=result)

    assert 'data-testid="delivery-date-available"' in html
    assert 'data-available="true"' in html
    assert "Wednesday, 10 June 2026" in html
    assert "Rs. 350" in html


def test_delivery_date_status_unavailable_partial() -> None:
    """Unavailable check_delivery result renders reason and next date."""
    result = CheckDeliveryOutput(
        city="Colombo 03",
        now="2026-06-08T10:00:00+05:30",
        checked_date="2026-06-09",
        available=False,
        rate=350.0,
        currency="LKR",
        reason="Sunday delivery unavailable",
        next_available_date="2026-06-10",
    )
    html = render_delivery_date_status(result=result)

    assert 'data-testid="delivery-date-unavailable"' in html
    assert 'data-available="false"' in html
    assert "Sunday delivery unavailable" in html
    assert "Tuesday, 9 June 2026" in html
    assert "Wednesday, 10 June 2026" in html


def test_delivery_date_error_partial() -> None:
    """Past date error partial is user-friendly and alert-marked."""
    html = render_delivery_date_error(
        title="Date in the past",
        message="Please choose today or a future date.",
    )

    assert 'data-testid="delivery-date-error"' in html
    assert 'role="alert"' in html
    assert "Date in the past" in html
    assert "Please choose today or a future date." in html
