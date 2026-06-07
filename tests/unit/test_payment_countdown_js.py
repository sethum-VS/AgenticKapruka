"""Tests for Alpine paymentCountdown static script."""

from __future__ import annotations

from pathlib import Path

PAYMENT_COUNTDOWN_JS = (
    Path(__file__).resolve().parent.parent.parent / "static" / "js" / "payment-countdown.js"
)


def test_payment_countdown_js_registers_alpine_component() -> None:
    """payment-countdown.js defines paymentCountdown with MM:SS tick and warning threshold."""
    source = PAYMENT_COUNTDOWN_JS.read_text()

    assert 'Alpine.data("paymentCountdown"' in source
    assert "remainingSeconds" in source
    assert "display" in source
    assert "expired" in source
    assert "warning" in source
    assert "600" in source
    assert "setInterval" in source
