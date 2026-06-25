"""Unit tests for graphs.state AgentState TypedDict."""

from __future__ import annotations

import operator
from typing import get_type_hints

from langchain_core.messages import HumanMessage

from graphs.state import (
    AgentState,
    CheckoutStep,
    CurrencyCode,
    Intent,
    ModelTier,
    ToolInvocation,
)
from lib.chat.intent_metadata import IntentMetadata


def test_agent_state_minimal_dict_passes_type_check() -> None:
    """Minimal AgentState dict satisfies the TypedDict schema."""
    state: AgentState = {
        "messages": [HumanMessage(content="birthday cake for mom")],
        "session_id": "sess-test-001",
    }
    assert state["session_id"] == "sess-test-001"
    assert len(state["messages"]) == 1


def test_agent_state_all_fields_optional_except_messages_reducer() -> None:
    """Every field is declared on AgentState with messages using operator.add."""
    hints = get_type_hints(AgentState, include_extras=True)
    expected_keys = {
        "messages",
        "intent",
        "intent_metadata",
        "hybrid_context",
        "tool_calls",
        "tool_results",
        "tool_call_count",
        "tool_trace",
        "agent_loop_done",
        "agent_loop_exit_reason",
        "agent_loop_iterations",
        "agent_clarifying_question",
        "agent_tool_error",
        "model_tier",
        "session_id",
        "zep_thread_id",
        "currency",
        "session_budget_max",
        "session_budget_currency",
        "session_delivery_city_canonical",
        "session_delivery_date",
        "session_product_focus",
        "session_search_query",
        "session_occasion",
        "session_recipient_hint",
        "session_awaiting_delivery_date",
        "session_awaiting_gift_preferences",
        "session_delivery_city_confirmed",
        "session_shipment_address_raw",
        "delivery_city_raw",
        "delivery_city_canonical",
        "delivery_city_status",
        "delivery_city_candidates",
        "delivery_date",
        "delivery_context_ready",
        "checkout_state",
        "response_html",
        "assistant_message",
        "zep_memory_facts",
        "last_search_products",
        "last_visible_products",
        "search_broaden_applied",
        "cart_action_result",
    }
    assert set(hints) == expected_keys

    messages_hint = hints["messages"]
    assert hasattr(messages_hint, "__metadata__")
    assert messages_hint.__metadata__[0] is operator.add


def test_agent_state_full_optional_fields() -> None:
    """Fully populated AgentState accepts all orchestration field values."""
    metadata: IntentMetadata = {
        "is_situational": False,
        "detected_vernacular": "en",
        "requires_delivery_validation": False,
        "target_city": None,
        "budget_max": None,
    }
    state: AgentState = {
        "messages": [],
        "intent": "discovery",
        "intent_metadata": metadata,
        "hybrid_context": {"categories": ["Birthday"]},
        "tool_calls": [{"name": "kapruka_search_products", "args": {"q": "cake"}}],
        "tool_results": {"kapruka_search_products": {"results": []}},
        "tool_call_count": 1,
        "model_tier": "flash",
        "session_id": "sess-abc",
        "zep_thread_id": "zep-thread-xyz",
        "currency": "LKR",
        "checkout_state": "cart",
        "response_html": "<div>Hello</div>",
    }
    intent: Intent | None = state["intent"]
    tier: ModelTier | None = state["model_tier"]
    currency: CurrencyCode | None = state["currency"]
    checkout: CheckoutStep | None = state["checkout_state"]
    assert intent == "discovery"
    assert tier == "flash"
    assert currency == "LKR"
    assert checkout == "cart"


def test_agent_state_agent_loop_fields_optional() -> None:
    """Agent loop trace fields accept minimal and populated optional values."""
    minimal: AgentState = {
        "messages": [HumanMessage(content="cakes for mom")],
        "session_id": "sess-loop-001",
    }
    assert minimal.get("tool_trace") is None
    assert minimal.get("agent_loop_done") is None
    assert minimal.get("agent_clarifying_question") is None

    invocation: ToolInvocation = {
        "name": "kapruka_search_products",
        "args": {"q": "birthday cake"},
        "result": {"results": [{"id": "cake001", "name": "Choc Cake"}]},
    }
    populated: AgentState = {
        "messages": [],
        "session_id": "sess-loop-002",
        "tool_trace": [invocation],
        "agent_loop_done": True,
        "agent_clarifying_question": "Who is the gift for?",
        "tool_results": {"kapruka_search_products": invocation["result"]},
        "tool_call_count": 1,
    }
    assert populated["tool_trace"] == [invocation]
    assert populated["agent_loop_done"] is True
    assert populated["agent_clarifying_question"] == "Who is the gift for?"
