"""Tests for Alpine rateLimitCountdown static script."""

from __future__ import annotations

from pathlib import Path

RATE_LIMIT_COUNTDOWN_JS = (
    Path(__file__).resolve().parent.parent.parent / "static" / "js" / "rate-limit-countdown.js"
)


def test_rate_limit_countdown_js_registers_alpine_component() -> None:
    """rate-limit-countdown.js defines rateLimitCountdown with auto-dismiss."""
    source = RATE_LIMIT_COUNTDOWN_JS.read_text()

    assert 'Alpine.data("rateLimitCountdown"' in source
    assert "remainingSeconds" in source
    assert "dismissed" in source
    assert "display" in source
    assert "setInterval" in source
    assert "destroy" in source
