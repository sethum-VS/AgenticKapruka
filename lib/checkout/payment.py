"""Checkout payment CTA context for click-to-pay countdown UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class PaymentCtaContext:
    """Fields rendered in templates/checkout/payment_cta.html."""

    checkout_url: str
    order_ref: str
    grand_total: float
    currency: str
    expires_at: str


def parse_expires_at_iso(value: str) -> datetime:
    """Parse Kapruka expires_at ISO 8601 timestamp."""
    return datetime.fromisoformat(value)


def countdown_remaining_seconds(expires_at: datetime, *, now: datetime) -> int:
    """Whole seconds remaining until expiry (zero when already expired)."""
    return max(0, int((expires_at - now).total_seconds()))


def format_countdown_mm_ss(seconds: int) -> str:
    """Format seconds as MM:SS for the payment link countdown."""
    minutes, secs = divmod(max(0, seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def is_countdown_warning(seconds: int) -> bool:
    """True when under ten minutes remain on the checkout link."""
    return 0 < seconds < 600


def payment_cta_from_finalize(
    *,
    checkout_url: str,
    order_ref: str,
    order_summary: dict[str, Any] | None,
    expires_at: str,
    currency: str = "LKR",
) -> PaymentCtaContext | None:
    """Build payment CTA context from kapruka_create_order finalize fields."""
    url = checkout_url.strip()
    ref = order_ref.strip()
    expiry = expires_at.strip()
    if not url or not ref or not expiry:
        return None

    summary_currency = currency
    grand_total = 0.0
    if isinstance(order_summary, dict):
        summary_currency = str(order_summary.get("currency") or currency)
        grand_total = float(order_summary.get("grand_total") or 0.0)

    return PaymentCtaContext(
        checkout_url=url,
        order_ref=ref,
        grand_total=grand_total,
        currency=summary_currency,
        expires_at=expiry,
    )
