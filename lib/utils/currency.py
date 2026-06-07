"""Currency display formatting for templates and UI."""

from __future__ import annotations

SUPPORTED_CURRENCIES = frozenset({"LKR", "USD", "GBP", "AUD", "CAD", "EUR"})

_SYMBOLS: dict[str, str] = {
    "LKR": "Rs.",
    "USD": "$",
    "GBP": "£",
    "AUD": "A$",
    "CAD": "C$",
    "EUR": "€",
}

_ZERO_DECIMAL_CURRENCIES = frozenset({"LKR"})


def format_currency(amount: float | int, currency: str = "LKR") -> str:
    """Format a monetary amount for display in the given ISO currency code."""
    code = currency.upper()
    if code not in SUPPORTED_CURRENCIES:
        msg = f"Unsupported currency: {currency}"
        raise ValueError(msg)

    symbol = _SYMBOLS[code]
    if code in _ZERO_DECIMAL_CURRENCIES:
        return f"{symbol} {round(float(amount)):,}"

    return f"{symbol}{float(amount):,.2f}"
