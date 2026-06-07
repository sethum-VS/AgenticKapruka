"""Tests for lib.utils.currency format_currency."""

from __future__ import annotations

import pytest

from lib.utils.currency import format_currency


def test_format_currency_lkr_without_decimals() -> None:
    assert format_currency(1500, "LKR") == "Rs. 1,500"


def test_format_currency_lkr_rounds_floats() -> None:
    assert format_currency(1500.6, "LKR") == "Rs. 1,501"
    assert format_currency(1500.4, "LKR") == "Rs. 1,500"


@pytest.mark.parametrize(
    ("amount", "currency", "expected"),
    [
        (1500, "USD", "$1,500.00"),
        (12.5, "USD", "$12.50"),
        (1500, "EUR", "€1,500.00"),
        (99.9, "GBP", "£99.90"),
        (2500, "AUD", "A$2,500.00"),
        (1800.25, "CAD", "C$1,800.25"),
    ],
)
def test_format_currency_decimal_currencies(
    amount: float,
    currency: str,
    expected: str,
) -> None:
    assert format_currency(amount, currency) == expected


def test_format_currency_accepts_lowercase_code() -> None:
    assert format_currency(1500, "lkr") == "Rs. 1,500"


def test_format_currency_rejects_unknown_code() -> None:
    with pytest.raises(ValueError, match="Unsupported currency"):
        format_currency(100, "JPY")
