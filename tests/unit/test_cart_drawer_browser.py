"""Browser verification for Alpine cart drawer open/close interactions."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, sync_playwright

from app.templating import render_cart_drawer
from lib.redis.cart import StoredCartItem

CART_DRAWER_JS = (
    Path(__file__).resolve().parent.parent.parent / "static" / "js" / "cart-drawer.js"
).read_text()
APP_CSS = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "app.css"


def _cart_drawer_harness_html(*, item_count: int = 0) -> str:
    items: list[StoredCartItem] = []
    if item_count > 0:
        items = [
            StoredCartItem(
                product_id="cake001",
                quantity=item_count,
                icing_text=None,
                name="Test Cake",
                price_amount=4500.0,
                price_currency="LKR",
            ),
        ]
    drawer_html = render_cart_drawer(items=items)
    css = APP_CSS.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>{css}</style>
    <script>{CART_DRAWER_JS}</script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
  </head>
  <body class="bg-commerce-cream p-4">
    {drawer_html}
  </body>
</html>"""


def _wait_for_alpine(page: Page) -> None:
    page.wait_for_function(
        """() => {
          const root = document.querySelector('[data-testid="cart-drawer"]');
          return window.Alpine && root?._x_dataStack;
        }"""
    )


def _drawer_open_state(page: Page) -> bool:
    return page.evaluate(
        """() => {
          const root = document.querySelector('[data-testid="cart-drawer"]');
          if (!root || !window.Alpine) return false;
          return Boolean(Alpine.$data(root)?.open);
        }"""
    )


@pytest.mark.browser
def test_cart_drawer_opens_and_closes_on_backdrop_click() -> None:
    """Clicking the cart icon opens the drawer; backdrop click closes it."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.set_content(_cart_drawer_harness_html())
        _wait_for_alpine(page)

        assert _drawer_open_state(page) is False

        page.locator('[data-testid="cart-icon"]').click()
        page.wait_for_function(
            """() => {
              const root = document.querySelector('[data-testid="cart-drawer"]');
              return Boolean(root && window.Alpine?.$data(root)?.open);
            }"""
        )

        page.evaluate(
            """() => {
              const backdrop = document.querySelector('[data-testid="cart-backdrop"]');
              backdrop?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
            }"""
        )
        page.wait_for_function(
            """() => {
              const root = document.querySelector('[data-testid="cart-drawer"]');
              return root && window.Alpine && !Alpine.$data(root)?.open;
            }"""
        )
        assert _drawer_open_state(page) is False

        browser.close()


@pytest.mark.browser
def test_cart_drawer_closes_on_escape_key() -> None:
    """Pressing Escape while the drawer is open closes it."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.set_content(_cart_drawer_harness_html(item_count=2))
        _wait_for_alpine(page)

        page.locator('[data-testid="cart-icon"]').click()
        page.wait_for_function(
            """() => {
              const root = document.querySelector('[data-testid="cart-drawer"]');
              return Boolean(root && window.Alpine?.$data(root)?.open);
            }"""
        )

        page.keyboard.press("Escape")
        page.wait_for_function(
            """() => {
              const root = document.querySelector('[data-testid="cart-drawer"]');
              return root && window.Alpine && !Alpine.$data(root)?.open;
            }"""
        )
        assert _drawer_open_state(page) is False

        browser.close()


@pytest.mark.browser
def test_cart_drawer_syncs_badge_from_htmx_cart_swap() -> None:
    """htmx:afterSwap on #cart-panel updates the Alpine badge count."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.set_content(_cart_drawer_harness_html())
        _wait_for_alpine(page)

        page.evaluate(
            """() => {
              const panel = document.getElementById('cart-panel');
              panel.setAttribute('data-item-count', '4');
              document.body.dispatchEvent(
                new CustomEvent('htmx:afterSwap', {
                  detail: { target: panel },
                  bubbles: true,
                })
              );
            }"""
        )

        page.wait_for_function(
            """() => {
              const root = document.querySelector('[data-testid="cart-drawer"]');
              return window.Alpine?.$data(root)?.itemCount === 4;
            }"""
        )
        badge_text = page.locator('[data-testid="cart-badge"]').text_content()
        assert badge_text == "4"

        browser.close()
