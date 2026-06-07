"""Browser verification for Alpine payment countdown expiry state."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, sync_playwright

from app.templating import render_payment_cta
from lib.checkout.payment import PaymentCtaContext

PAYMENT_COUNTDOWN_JS = (
    Path(__file__).resolve().parent.parent.parent / "static" / "js" / "payment-countdown.js"
).read_text()
APP_CSS = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "app.css"


def _payment_cta_harness_html(*, expires_at: str) -> str:
    payment_html = render_payment_cta(
        payment=PaymentCtaContext(
            checkout_url="https://www.kapruka.com/checkout/pay/abc123",
            order_ref="ORD-20260608-7823",
            grand_total=9350.0,
            currency="LKR",
            expires_at=expires_at,
        ),
    )
    css = APP_CSS.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>{css}</style>
    <script>{PAYMENT_COUNTDOWN_JS}</script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
  </head>
  <body class="bg-commerce-cream p-4">
    {payment_html}
  </body>
</html>"""


def _wait_for_alpine(page: Page) -> None:
    page.wait_for_function(
        """() => {
          const root = document.querySelector('[data-testid="checkout-payment-cta"]');
          return window.Alpine && root?._x_dataStack;
        }"""
    )


def _countdown_state(page: Page) -> dict[str, object]:
    return page.evaluate(
        """() => {
          const root = document.querySelector('[data-testid="checkout-payment-cta"]');
          if (!root || !window.Alpine) return {};
          const data = Alpine.$data(root);
          return {
            expired: data.expired,
            warning: data.warning,
            display: data.display,
          };
        }"""
    )


@pytest.mark.browser
def test_payment_countdown_shows_warning_under_ten_minutes() -> None:
    """Countdown enters warning state when fewer than ten minutes remain."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    expires_at = (datetime.now(ZoneInfo("Asia/Colombo")) + timedelta(minutes=5)).isoformat()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.set_content(_payment_cta_harness_html(expires_at=expires_at))
        _wait_for_alpine(page)

        state = _countdown_state(page)
        assert state.get("warning") is True
        assert state.get("expired") is False
        browser.close()


@pytest.mark.browser
def test_payment_countdown_shows_expired_message_at_zero() -> None:
    """Expired checkout links show the expired message and 00:00 countdown."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.set_content(_payment_cta_harness_html(expires_at="2020-01-01T00:00:00+05:30"))
        _wait_for_alpine(page)

        state = _countdown_state(page)
        assert state.get("expired") is True
        assert state.get("display") == "00:00"

        expired = page.locator('[data-testid="checkout-payment-expired"]')
        assert expired.is_visible()
        assert "expired" in expired.inner_text().lower()
        browser.close()
