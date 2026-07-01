"""Timezone helpers for Kapruka delivery (Asia/Colombo)."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

COLOMBO_TZ = ZoneInfo("Asia/Colombo")


def colombo_now() -> datetime:
    """Current wall-clock time in Asia/Colombo."""
    return datetime.now(COLOMBO_TZ)


def colombo_today() -> date:
    """Today's calendar date in Asia/Colombo."""
    return colombo_now().date()


def colombo_today_iso() -> str:
    """Today's date as YYYY-MM-DD in Asia/Colombo."""
    return colombo_today().isoformat()


def format_delivery_date_friendly(iso_date: str) -> str:
    """Format YYYY-MM-DD as a human-friendly date (e.g. Sunday, 28 June 2026)."""
    parsed = date.fromisoformat(iso_date)
    return f"{parsed.strftime('%A')}, {parsed.day} {parsed.strftime('%B %Y')}"


def is_past_colombo_date(delivery_date: str) -> bool:
    """True when delivery_date (YYYY-MM-DD) is before today in Asia/Colombo."""
    return date.fromisoformat(delivery_date) < colombo_today()
