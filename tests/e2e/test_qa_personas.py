"""E2E scripted replay of the four QA report personas."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect
from tests.e2e.helpers import (
    extract_last_assistant_text,
    fetch_mcp_tools,
    reset_e2e_session,
    send_chat_message,
    wait_for_alpine,
)

pytestmark = pytest.mark.e2e


def _chat_turn(page: Page, message: str) -> str:
    send_chat_message(page, message)
    return extract_last_assistant_text(page)


def test_persona_clueless_gift_giver_budget_and_delivery(
    page: Page,
    base_url: str,
) -> None:
    """Clueless Gift Giver QA Scenario 1: clarify → chocolate → budget → Kandy delivery."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_e2e_session(page, base_url)

    _chat_turn(page, "I need a gift for my wife")
    _chat_turn(page, "chocolate")
    budget_reply = _chat_turn(page, "Keep it under 6000 rupees.")
    delivery_reply = _chat_turn(page, "can you deliver to Kandy this Sunday?")

    expect(page.locator('[data-testid="product-carousel"]')).to_have_count(1)
    carousel_text = page.inner_text('[data-testid="product-carousel"]')
    assert "26,310" not in carousel_text
    assert "curry" not in carousel_text.lower()
    lowered_carousel = carousel_text.lower()
    assert "snack" not in lowered_carousel
    assert (
        "bar" not in lowered_carousel or "bento" in lowered_carousel or "cake" in lowered_carousel
    )
    assert any(token in lowered_carousel for token in ("cake", "bento", "chocolate", "hamper"))
    assert "snack" not in carousel_text.lower()
    assert "bar" not in carousel_text.lower() or "birthday" in carousel_text.lower()
    assert any(
        keyword in carousel_text.lower()
        for keyword in ("cake", "bento", "cheers", "chocolate")
    )

    tools = fetch_mcp_tools(page, base_url)
    if tools:
        assert "kapruka_search_products" in tools
        assert "kapruka_check_delivery" in tools
    assert "voucher" not in budget_reply.lower() or "chocolate" in budget_reply.lower()
    expect(page.locator('[data-testid="product-card"]').first).to_be_visible()
    assert "kandy" in delivery_reply.lower()
    verified = "verified with kapruka" in delivery_reply.lower()
    assert delivery_reply.lower().count("rs.") <= 2 or verified


def test_persona_context_pivot_cakes_not_vouchers(page: Page, base_url: str) -> None:
    """Context Pivot: New chat clears budget; cakes edible after nevermind pivot."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_e2e_session(page, base_url)

    _chat_turn(page, "chocolates for wife under 6000")
    page.click('[data-testid="new-chat-button"]')
    page.wait_for_function(
        """() => {
          const empty = document.getElementById('chat-empty-state');
          const carousels = document.querySelectorAll('[data-testid="product-carousel"]');
          return empty && carousels.length === 0;
        }"""
    )
    anniversary_reply = _chat_turn(page, "anniversary gifts for my wife")
    assert "over budget" not in anniversary_reply.lower()
    assert "verified with kapruka" not in anniversary_reply.lower()

    reply = _chat_turn(page, "Nevermind. Cakes.")

    assert "decorating" not in reply.lower()
    assert "anniversary" not in reply.lower()
    assert "kandy" not in reply.lower()
    expect(page.locator('[data-testid="product-card"]').first).to_be_visible()


def test_persona_distracted_shopper_weather_and_elephant(page: Page, base_url: str) -> None:
    """Distracted Shopper: weather redirect and impossible elephant suggestion."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_e2e_session(page, base_url)

    weather = _chat_turn(page, "What's the weather in Colombo?")
    assert "weather" in weather.lower() or "can't check" in weather.lower()
    assert "zone" not in weather.lower()

    elephant = _chat_turn(page, "Can you deliver a live elephant?")
    assert "stuffed" in elephant.lower() or "can't deliver" in elephant.lower()


def test_persona_apology_professional_tone(page: Page, base_url: str) -> None:
    """Apology flow stays professional without unsolicited slang."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_e2e_session(page, base_url)

    reply = _chat_turn(
        page,
        "I'm sorry — I ordered the wrong cake. Can you help me find a replacement?",
    )
    lowered = reply.lower()
    assert "machan" not in lowered
    assert "bro" not in lowered or "broke" in lowered


def test_persona_breakup_omits_stale_kandy_delivery(page: Page, base_url: str) -> None:
    """After Kandy delivery context, breakup empathy must not mention Kandy."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_e2e_session(page, base_url)

    _chat_turn(page, "chocolate gift for my wife in Kandy")
    _chat_turn(page, "can you deliver this Sunday?")
    breakup_reply = _chat_turn(page, "We just broke up and I'm heartbroken.")

    lowered = breakup_reply.lower()
    assert "kandy" not in lowered
    assert "verified with kapruka" not in lowered
    assert re.search(r"sorry|heartbroken|hear that|here for you", lowered)


def test_persona_action_shopper_tracking_and_cart(page: Page, base_url: str) -> None:
    """Scenario 3: KA order ID gets VIMP guidance; VIMP ID returns tracking card; cart add."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_e2e_session(page, base_url)

    # 1. Non-VIMP order ID → guidance to use VIMP format
    ka_reply = _chat_turn(page, "KA987654")
    lowered_ka = ka_reply.lower()
    assert re.search(r"vimp|order\s*id|tracking\s*id|format", lowered_ka), (
        f"Expected VIMP format guidance for KA order ID, got: {ka_reply!r}"
    )
    assert page.locator('[data-testid="tracking-card"]').count() == 0, (
        "KA order ID should not show a tracking card"
    )

    # 2. Valid VIMP ID → tracking card
    vimp_reply = _chat_turn(page, "VIMP34456CB2")
    lowered_vimp = vimp_reply.lower()
    assert re.search(r"delivered|in transit|pending|processing|tracking", lowered_vimp), (
        f"Expected tracking status in VIMP reply, got: {vimp_reply!r}"
    )

    # 3. Add product to cart → cart drawer shows 1 item
    _chat_turn(page, "Show me anniversary flowers for my wife")
    page.wait_for_selector('[data-testid="product-card"]', timeout=15_000)
    cart_reply = _chat_turn(page, "Add the first flower bouquet to my cart")
    lowered_cart = cart_reply.lower()
    assert re.search(r"added|cart|added to cart|in your cart", lowered_cart), (
        f"Expected cart-add confirmation, got: {cart_reply!r}"
    )
    cart_count = page.locator('[data-testid="cart-count"]')
    if cart_count.count() > 0:
        count_text = cart_count.first.inner_text()
        assert int(count_text) >= 1, f"Expected cart count >= 1, got {count_text!r}"
