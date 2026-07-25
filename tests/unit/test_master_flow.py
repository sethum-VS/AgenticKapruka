"""Unit tests for master flow supervisor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graphs.state import AgentState
from lib.chat.master_flow import (
    MasterFlowAlignment,
    apply_master_flow_alignment,
    infer_active_flow,
    invoke_master_flow_llm,
    message_matches_checkout_exit,
    should_invoke_master_flow,
)
from lib.chat.routing import route_after_master_flow


def _state(**kwargs: object) -> AgentState:
    base: dict[str, object] = {
        "messages": [HumanMessage(content="hello")],
        "intent": "discovery",
    }
    base.update(kwargs)
    return base  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"checkout_state": "delivery_city"}, "checkout_active"),
        ({"session_awaiting_delivery_date": True}, "awaiting_delivery_date"),
        (
            {"session_awaiting_clarification_dimension": "product"},
            "awaiting_clarification",
        ),
        ({"last_visible_products": [{"id": "p1"}]}, "carousel_context"),
        (
            {
                "delivery_context_ready": False,
                "intent_metadata": {"target_city": "Colombo"},
            },
            "delivery_resolution",
        ),
        ({}, "free_discovery"),
    ],
)
def test_infer_active_flow(fields: dict[str, object], expected: str) -> None:
    assert infer_active_flow(_state(**fields)) == expected


def test_should_invoke_awaiting_delivery_date_without_date() -> None:
    state = _state(
        session_awaiting_delivery_date=True,
        messages=[HumanMessage(content="ok sounds good")],
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is True
    assert reason == "awaiting_delivery_date_without_parseable_date"


def test_should_not_invoke_bare_category_pivot_with_stale_carousel() -> None:
    """Fresh flowers / cakes pivots must search — not Flash-clarify over stale carousel."""
    state = _state(
        intent_metadata={"topic_pivot": True},
        last_visible_products=[{"id": "old-cake"}],
        session_search_query="anniversary cake",
        messages=[HumanMessage(content="What about just normal fresh flowers?")],
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is False
    assert reason == "bare_category_pivot_fast_path"


def test_should_invoke_awaiting_clarification_unanswered() -> None:
    state = _state(
        session_awaiting_clarification_dimension="budget",
        messages=[HumanMessage(content="red roses please")],
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is True
    assert reason == "awaiting_clarification_dimension_unanswered"


def test_should_invoke_checkout_discovery_conflict() -> None:
    state = _state(
        checkout_state="delivery_city",
        intent="discovery",
        messages=[HumanMessage(content="what cakes do you have?")],
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is True
    assert reason == "checkout_active_with_discovery_intent"


def test_should_invoke_delivery_only_product_search_conflict() -> None:
    state = _state(
        intent="discovery",
        messages=[HumanMessage(content="delivery fee to Colombo on 2026-07-05")],
        intent_metadata={
            "requires_delivery_validation": True,
            "target_city": "Colombo",
            "delivery_date": "2026-07-05",
        },
        last_visible_products=[{"id": "stale"}],
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is True
    assert reason == "delivery_only_with_stale_carousel"


def test_should_invoke_topic_pivot_stale_carousel() -> None:
    state = _state(
        intent_metadata={"topic_pivot": True},
        last_visible_products=[{"id": "old"}],
        session_search_query="chocolate",
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is True
    assert reason == "topic_pivot_with_stale_carousel"


def test_should_invoke_long_session_drift() -> None:
    messages = [HumanMessage(content=f"turn {i}") for i in range(8)]
    # Recipient switch with stale carousel (no budget refine) still triggers drift.
    messages.append(HumanMessage(content="actually for my sister"))
    state = _state(
        messages=messages,
        last_visible_products=[{"id": "stale"}],
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is True
    assert reason == "long_session_drift"


def test_should_not_invoke_budget_refinement_long_session() -> None:
    """Budget-only refine must skip master-flow even after many turns."""
    messages = [HumanMessage(content=f"turn {i}") for i in range(8)]
    messages.append(HumanMessage(content="under 6000"))
    state = _state(
        messages=messages,
        last_visible_products=[{"id": "chocolate-1", "price": 5000}],
        session_product_focus="chocolate",
        session_budget_max=6000,
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is False
    assert reason == "budget_refinement_fast_path"


def test_should_not_invoke_bare_category_normal_fresh_flowers() -> None:
    state = _state(
        messages=[HumanMessage(content="Normal fresh flowers?")],
        last_visible_products=[{"id": "cake-1"}],
        session_product_focus="cake",
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is False
    assert reason == "bare_category_pivot_fast_path"


def test_should_not_invoke_when_specificity_band_is_clarify() -> None:
    state = _state(
        specificity_band="clarify",
        agent_clarifying_question="Who is the gift for?",
        checkout_state="delivery_city",
        intent="discovery",
    )
    invoke, reason = should_invoke_master_flow(state)
    assert invoke is False
    assert reason == "specificity_clarify_fast_path"


def test_should_not_invoke_when_feature_disabled() -> None:
    state = _state(checkout_state="delivery_city", intent="discovery")
    with patch("lib.chat.master_flow.get_settings") as mock_settings:
        mock_settings.return_value.master_flow_enabled = False
        invoke, reason = should_invoke_master_flow(state)
    assert invoke is False
    assert reason == "feature_disabled"


def test_apply_master_flow_clarify_above_threshold() -> None:
    alignment = MasterFlowAlignment(
        decision="clarify",
        confidence=0.9,
        active_flow="awaiting_delivery_date",
        clarifying_question="Which delivery date works for you?",
    )
    updates = apply_master_flow_alignment(
        _state(),
        alignment,
        user_message="show cakes",
    )
    assert updates["master_clarifying_question"] == "Which delivery date works for you?"


def test_apply_master_flow_no_op_below_threshold() -> None:
    alignment = MasterFlowAlignment(
        decision="clarify",
        confidence=0.5,
        active_flow="free_discovery",
        clarifying_question="What gift?",
        context_reset=True,
    )
    updates = apply_master_flow_alignment(
        _state(last_visible_products=[{"id": "x"}]),
        alignment,
        user_message="wife budget 5000",
    )
    assert "master_clarifying_question" not in updates
    assert "last_visible_products" not in updates


def test_checkout_exit_blocked_without_allowlist() -> None:
    alignment = MasterFlowAlignment(
        decision="checkout_exit",
        confidence=0.95,
        active_flow="checkout_active",
        checkout_action="exit",
    )
    updates = apply_master_flow_alignment(
        _state(checkout_state="delivery_city"),
        alignment,
        user_message="what cakes do you have?",
    )
    assert "checkout_state" not in updates


def test_checkout_exit_allowed_with_allowlist() -> None:
    alignment = MasterFlowAlignment(
        decision="checkout_exit",
        confidence=0.95,
        active_flow="checkout_active",
        checkout_action="exit",
    )
    updates = apply_master_flow_alignment(
        _state(checkout_state="delivery_city"),
        alignment,
        user_message="cancel checkout",
    )
    assert updates.get("checkout_state") is None
    assert message_matches_checkout_exit("cancel checkout")


def test_apply_context_reset_clears_carousel() -> None:
    alignment = MasterFlowAlignment(
        decision="pivot",
        confidence=0.9,
        active_flow="carousel_context",
        context_reset=True,
    )
    updates = apply_master_flow_alignment(
        _state(last_visible_products=[{"id": "x"}], session_search_query="roses"),
        alignment,
        user_message="wife budget 5000",
    )
    assert updates.get("last_visible_products") is None
    assert updates.get("session_search_query") is None


@pytest.mark.asyncio
async def test_invoke_master_flow_llm_fail_open() -> None:
    result = await invoke_master_flow_llm(
        _state(),
        active_flow="free_discovery",
        trigger_reason="test",
        genai_client=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_invoke_master_flow_llm_parses_response() -> None:
    alignment = MasterFlowAlignment(
        decision="proceed",
        confidence=0.8,
        active_flow="free_discovery",
    )

    with patch(
        "lib.chat.master_flow.generate_content",
        return_value=alignment,
    ) as mock_generate:
        result = await invoke_master_flow_llm(
            _state(),
            active_flow="free_discovery",
            trigger_reason="test",
            genai_client=object(),
        )
    assert result is not None
    assert result.decision == "proceed"
    mock_generate.assert_awaited_once()


def test_route_after_master_flow_clarify_short_circuit() -> None:
    state = _state(master_clarifying_question="Which date?")
    assert route_after_master_flow(state) == "generate_response"


def test_route_after_master_flow_proceed_delegates() -> None:
    state = _state(intent="checkout")
    assert route_after_master_flow(state) == "run_checkout_graph"
