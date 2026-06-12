"""Unit tests for shopping graph state helpers."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from graphs.shopping_graph import append_message_state


def test_append_message_state_resets_per_turn_agent_fields() -> None:
    """Follow-up deltas must not carry prior tool_trace or clarifying question."""
    delta = append_message_state("cakes")

    assert delta["messages"][-1].content == "cakes"
    assert delta["tool_trace"] == []
    assert delta["tool_results"] == {}
    assert delta["tool_call_count"] == 0
    assert delta["agent_clarifying_question"] is None
    assert delta["agent_tool_error"] is None
    assert delta["agent_loop_done"] is None
    assert delta["agent_loop_exit_reason"] is None
    assert delta["agent_loop_iterations"] is None


def test_append_message_state_passes_currency_when_set() -> None:
    """Currency override still applies on follow-up turns."""
    delta = append_message_state("cakes", currency="USD")

    assert isinstance(delta["messages"][0], HumanMessage)
    assert delta["currency"] == "USD"
