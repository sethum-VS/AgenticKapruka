"""Unit tests for graphs.state AgentState TypedDict."""

from __future__ import annotations

import operator
from typing import get_type_hints

from langchain_core.messages import HumanMessage

from graphs.state import AgentState, CheckoutStep, CurrencyCode, Intent, ModelTier
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
        "model_tier",
        "session_id",
        "zep_thread_id",
        "currency",
        "checkout_state",
        "response_html",
        "assistant_message",
        "zep_memory_facts",
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
