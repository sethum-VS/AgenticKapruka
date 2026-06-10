"""HybridRAG E2E: DOM extraction, MCP tool alignment, and LLM-judge rubrics."""

from __future__ import annotations

import pytest
from evals.llm_judge import (
    score_constraint_fidelity,
    score_mcp_tool_alignment,
    score_visual_fidelity,
)
from playwright.sync_api import Page, expect
from tests.e2e.helpers import (
    extract_chat_messages_html,
    fetch_mcp_tools,
    reset_mcp_log,
    send_chat_message,
    wait_for_alpine,
)

pytestmark = pytest.mark.e2e


def _assert_turn_fidelity(
    response_html: str,
    called_tools: list[str],
    expected_tools: list[str],
    *,
    must_contain: list[str] | None = None,
    require_product_ui: bool = False,
) -> None:
    tool_score = score_mcp_tool_alignment(called_tools, expected_tools)
    visual_score = score_visual_fidelity(
        response_html,
        require_product_card=require_product_ui,
        require_carousel=require_product_ui,
    )
    constraint_score = (
        score_constraint_fidelity(response_html, must_contain=must_contain)
        if must_contain
        else None
    )
    assert tool_score.verdict == "pass", tool_score.reason
    assert visual_score.verdict == "pass", visual_score.reason
    if constraint_score is not None:
        assert constraint_score.verdict == "pass", constraint_score.reason


def test_discovery_renders_product_card_and_calls_search(page: Page, base_url: str) -> None:
    """Output fidelity: search turn must call kapruka_search_products and render cards."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_mcp_log(page, base_url)

    send_chat_message(page, "Show me birthday cakes for my mom")
    page.wait_for_selector('[data-testid="product-card"]', timeout=60_000)

    html = extract_chat_messages_html(page)
    tools = fetch_mcp_tools(page, base_url)

    _assert_turn_fidelity(
        html,
        tools,
        ["kapruka_search_products"],
        must_contain=["Chocolate Birthday Cake"],
        require_product_ui=True,
    )
    expect(page.locator('[data-testid="product-carousel"]')).to_be_visible()


def test_check_delivery_endpoint_calls_mcp(page: Page, base_url: str) -> None:
    """Input alignment: /checkout/check-delivery must invoke kapruka_check_delivery for Kandy."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_mcp_log(page, base_url)

    response = page.request.post(
        f"{base_url}/checkout/check-delivery",
        form={"city": "Kandy", "delivery_date": "2026-12-25"},
    )
    assert response.ok
    html = response.text()
    tools = fetch_mcp_tools(page, base_url)

    tool_score = score_mcp_tool_alignment(tools, ["kapruka_check_delivery"])
    constraint_score = score_constraint_fidelity(
        html,
        must_contain=["Delivery available", "2026-12-25"],
    )
    assert tool_score.verdict == "pass", tool_score.reason
    assert constraint_score.verdict == "pass", constraint_score.reason
    assert 'data-testid="delivery-date-available"' in html


def test_tracking_renders_status_without_product_wall(page: Page, base_url: str) -> None:
    """Tracking intent calls kapruka_track_order and returns structured status UI."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_mcp_log(page, base_url)

    send_chat_message(page, "Track order VIMP34456CB2")

    html = extract_chat_messages_html(page)
    tools = fetch_mcp_tools(page, base_url)

    tool_score = score_mcp_tool_alignment(tools, ["kapruka_track_order"])
    visual_score = score_visual_fidelity(html, require_product_card=False)
    assert tool_score.verdict == "pass", tool_score.reason
    assert visual_score.verdict == "pass", visual_score.reason
    assert "VIMP34456CB2" in html or "Out for Delivery" in html
