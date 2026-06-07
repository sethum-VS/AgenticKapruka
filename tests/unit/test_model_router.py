"""Unit tests for graphs.model_router."""

from __future__ import annotations

from graphs.model_router import FLASH_MODEL, PRO_MODEL, select_model, select_model_tier
from graphs.state import AgentState


def _state(**overrides: object) -> AgentState:
    base: AgentState = {
        "messages": [],
        "session_id": "sess-router-001",
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def test_select_model_defaults_to_flash() -> None:
    assert select_model(_state()) == FLASH_MODEL
    assert select_model_tier(_state()) == "flash"


def test_select_model_escalates_on_checkout_review() -> None:
    state = _state(checkout_state="review")
    assert select_model_tier(state) == "pro"
    assert select_model(state) == PRO_MODEL


def test_select_model_respects_explicit_pro_tier() -> None:
    state = _state(model_tier="pro")
    assert select_model(state) == PRO_MODEL


def test_select_model_stays_flash_at_tool_call_threshold() -> None:
    state = _state(tool_call_count=3)
    assert select_model_tier(state) == "flash"
    assert select_model(state) == FLASH_MODEL


def test_select_model_escalates_after_fourth_tool_call() -> None:
    """tool_call_count > 3 triggers Pro escalation (4th call onward)."""
    state = _state(tool_call_count=4)
    assert select_model_tier(state) == "pro"
    assert select_model(state) == PRO_MODEL


def test_select_model_escalates_when_tool_call_count_exceeds_threshold() -> None:
    state = _state(tool_call_count=10)
    assert select_model(state) == PRO_MODEL
