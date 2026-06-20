"""Playwright E2E smoke tests: chat search, cart drawer, checkout form (no create_order)."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from tests.fixtures.mcp_mock import SEARCH_PRODUCTS_JSON

from lib.kapruka.tools.create_order import TOOL_NAME as CREATE_ORDER_TOOL

pytestmark = pytest.mark.e2e

_SEARCH_MESSAGE = "birthday cake"
_MOCK_PRODUCT_NAME = SEARCH_PRODUCTS_JSON["results"][0]["name"]
_MOCK_PRODUCT_ID = SEARCH_PRODUCTS_JSON["results"][0]["id"]


def _wait_for_alpine(page: Page) -> None:
    page.wait_for_function("() => Boolean(window.Alpine)")


def _search_and_wait_for_product_card(page: Page) -> None:
    page.fill("#chat-message", _SEARCH_MESSAGE)
    page.click('button[type="submit"]')
    page.wait_for_selector('[data-testid="product-card"]', timeout=60_000)


def test_chat_page_loads(page: Page, base_url: str) -> None:
    """GET /chat renders the chat shell and message form."""
    page.goto(f"{base_url}/chat")
    _wait_for_alpine(page)

    expect(page.locator("#chat-form")).to_be_visible()
    expect(page.locator("#chat-message")).to_be_visible()
    expect(page.get_by_role("heading", name="Kapruka Gift Assistant")).to_be_visible()
    expect(page.locator("#chat-empty-state")).to_be_visible()


def test_search_message_shows_product_card(page: Page, base_url: str) -> None:
    """Submitting a discovery query streams a product card into the chat."""
    page.goto(f"{base_url}/chat")
    _wait_for_alpine(page)
    _search_and_wait_for_product_card(page)

    card = page.locator(f'[data-product-id="{_MOCK_PRODUCT_ID}"]')
    expect(card).to_be_visible()
    expect(card).to_contain_text(_MOCK_PRODUCT_NAME)
    expect(page.locator('[data-testid="product-carousel"]')).to_be_visible()


def test_add_to_cart_shows_item_in_drawer(page: Page, base_url: str) -> None:
    """Add-to-cart HTMX swap updates the cart drawer badge and line item."""
    page.set_viewport_size({"width": 1280, "height": 800})
    page.goto(f"{base_url}/chat")
    _wait_for_alpine(page)
    _search_and_wait_for_product_card(page)

    page.locator('[data-testid="product-card"]').first.get_by_role(
        "button", name="Add to cart"
    ).click()
    page.wait_for_selector('[data-testid="cart-badge"]', state="visible", timeout=15_000)

    page.locator('[data-testid="cart-icon"]').click()
    drawer = page.locator('[data-testid="cart-drawer-panel"]')
    line_item = drawer.locator('[data-testid="cart-line-item"]')
    line_item.wait_for(state="visible", timeout=10_000)

    expect(line_item).to_be_visible()
    expect(line_item).to_contain_text(_MOCK_PRODUCT_NAME)
    expect(page.locator('[data-testid="cart-badge"]')).to_have_text("1")


def test_checkout_delivery_form_renders_without_create_order(page: Page, base_url: str) -> None:
    """Proceed to checkout and validate delivery form; mock MCP never calls create_order."""
    page.goto(f"{base_url}/chat")
    _wait_for_alpine(page)
    _search_and_wait_for_product_card(page)

    page.locator('[data-testid="product-card"]').first.get_by_role(
        "button", name="Add to cart"
    ).click()
    page.wait_for_selector('[data-testid="cart-badge"]', state="visible", timeout=15_000)

    page.locator('[data-testid="cart-icon"]').click()
    page.wait_for_selector('[data-testid="cart-proceed-checkout"]', timeout=10_000)
    page.locator('[data-testid="cart-proceed-checkout"]').click()

    page.wait_for_selector('[aria-label="Assistant message"]', timeout=60_000)
    assistant = page.locator('[aria-label="Assistant message"]').last
    expect(assistant).to_contain_text("checkout", ignore_case=True)

    response = page.request.post(
        f"{base_url}/checkout/validate-delivery",
        form={
            "address": "42 Lotus Road",
            "city": "Colombo 03",
            "location_type": "house",
            "date": "2026-12-25",
            "instructions": "Ring the bell",
        },
    )
    assert response.ok
    body = response.text()
    assert 'data-testid="delivery-form"' in body
    assert 'data-testid="delivery-form-valid"' in body
    assert 'data-testid="checkout-payment-cta"' not in body

    mcp_calls = page.request.get(f"{base_url}/e2e/mcp-calls").json()
    tools = mcp_calls.get("tools", [])
    assert CREATE_ORDER_TOOL not in tools
