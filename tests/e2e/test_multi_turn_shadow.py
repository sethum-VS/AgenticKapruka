"""Shadow tests: parametrized multi-turn transcripts with intent preservation."""

from __future__ import annotations

import pytest
from evals.llm_judge import (
    extract_html_text,
    score_intent_preservation,
    score_local_flavor,
    score_mcp_tool_alignment,
    score_visual_fidelity,
)
from evals.multi_turn_dataset import MultiTurnCase, get_multi_turn_dataset
from playwright.sync_api import Page
from tests.e2e.helpers import (
    extract_chat_messages_html,
    fetch_mcp_tools,
    reset_mcp_log,
    send_chat_message,
    wait_for_alpine,
)

pytestmark = pytest.mark.e2e


def _case_ids() -> list[str]:
    return [case.id for case in get_multi_turn_dataset().cases]


@pytest.fixture(params=_case_ids(), ids=_case_ids())
def multi_turn_case(request: pytest.FixtureRequest) -> MultiTurnCase:
    dataset = get_multi_turn_dataset()
    return next(case for case in dataset.cases if case.id == request.param)


def test_multi_turn_shadow_preserves_intent(
    page: Page,
    base_url: str,
    multi_turn_case: MultiTurnCase,
) -> None:
    """Run each golden transcript; judge final DOM + cumulative MCP calls."""
    page.goto(f"{base_url}/chat")
    wait_for_alpine(page)
    reset_mcp_log(page, base_url)

    for step in multi_turn_case.turns:
        send_chat_message(page, step.content)
        if multi_turn_case.final_expect_product_ui and step is multi_turn_case.turns[-1]:
            page.wait_for_selector('[data-testid="product-card"]', timeout=60_000)
        if multi_turn_case.final_expect_checkout_ui and step is multi_turn_case.turns[-1]:
            page.wait_for_selector('[aria-label="Assistant message"]', timeout=60_000)

    final_html = extract_chat_messages_html(page)
    tools = fetch_mcp_tools(page, base_url)

    tool_score = score_mcp_tool_alignment(
        tools,
        multi_turn_case.constraints.expected_tools_any_turn,
        threshold=0.5,
    )
    assert tool_score.verdict == "pass", (
        f"{multi_turn_case.id}: {tool_score.reason} (tools={tools})"
    )

    intent_score = score_intent_preservation(
        final_html,
        multi_turn_case.constraints.preserved_terms,
    )
    assert intent_score.verdict == "pass", f"{multi_turn_case.id}: {intent_score.reason}"

    if multi_turn_case.constraints.must_not_contain_in_final:
        lowered = final_html.lower()
        for forbidden in multi_turn_case.constraints.must_not_contain_in_final:
            assert forbidden.lower() not in lowered, (
                f"{multi_turn_case.id}: forbidden term {forbidden!r} in final HTML"
            )

    if multi_turn_case.final_expect_product_ui:
        visual = score_visual_fidelity(
            final_html,
            require_product_card=True,
            require_carousel=True,
        )
        assert visual.verdict == "pass", f"{multi_turn_case.id}: {visual.reason}"

    if multi_turn_case.query_mode == "situational":
        final_text = extract_html_text(final_html)
        local_flavor_score = score_local_flavor(
            final_text,
            query_mode="situational",
            threshold=0.75,
        )
        assert local_flavor_score.score >= 0.75, (
            f"Response failed local flavor alignment gate. Score: {local_flavor_score.score}"
        )
        lowered = final_text.lower()
        warm_curation = any(
            marker in lowered
            for marker in (
                "curated",
                "thoughtful",
                "gentle",
                "aiyo",
                "machan",
                "hodata",
            )
        )
        assert warm_curation, (
            f"{multi_turn_case.id}: situational reply should show warm curation tone"
        )
