"""E2E scripted replay of the four QA report personas."""

from __future__ import annotations

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
    """Clueless Gift Giver: chocolate thread, budget refine, Kandy delivery."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_e2e_session(page, base_url)

    _chat_turn(page, "chocolate gift for my wife in Kandy")
    _chat_turn(page, "under 6000")
    reply = _chat_turn(page, "can you deliver this Sunday?")

    tools = fetch_mcp_tools(page, base_url)
    assert "kapruka_search_products" in tools
    assert "kapruka_check_delivery" in tools
    assert "voucher" not in reply.lower() or "chocolate" in reply.lower()
    expect(page.locator('[data-testid="product-card"]').first).to_be_visible()


def test_persona_context_pivot_cakes_not_vouchers(page: Page, base_url: str) -> None:
    """Context Pivot: cakes edible after nevermind pivot without voucher pivot."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_e2e_session(page, base_url)

    _chat_turn(page, "chocolates for wife under 6000")
    reply = _chat_turn(page, "Nevermind. Cakes.")

    assert "decorating" not in reply.lower()
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
