"""LangGraph AgentState schema for the shopping assistant."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage

from lib.chat.intent_metadata import IntentMetadata

Intent = Literal["discovery", "checkout", "tracking", "general"]
ModelTier = Literal["flash", "pro"]
AgentLoopExitReason = Literal[
    "finish",
    "ask_user",
    "max_iterations",
    "duplicate_guard",
]
CheckoutStep = Literal[
    "cart",
    "delivery_city",
    "delivery_date",
    "recipient",
    "sender",
    "review",
    "finalize",
]
CurrencyCode = Literal["LKR", "USD", "GBP", "AUD", "CAD", "EUR"]


class ToolInvocation(TypedDict):
    """Single MCP tool call recorded in the bounded agent loop trace."""

    name: str
    args: dict[str, Any]
    result: Any


class AgentState(TypedDict):
    """Shared state for the main shopping LangGraph."""

    messages: Annotated[list[BaseMessage], operator.add]
    intent: Intent | None
    intent_metadata: IntentMetadata | None
    hybrid_context: dict[str, Any] | None
    tool_calls: list[dict[str, Any]] | None
    tool_results: dict[str, Any] | None
    tool_call_count: int | None
    tool_trace: list[ToolInvocation] | None
    agent_loop_done: bool | None
    agent_loop_exit_reason: AgentLoopExitReason | None
    agent_loop_iterations: int | None
    agent_clarifying_question: str | None
    model_tier: ModelTier | None
    session_id: str | None
    zep_thread_id: str | None
    currency: CurrencyCode | None
    checkout_state: CheckoutStep | None
    response_html: str | None
    assistant_message: str | None
    zep_memory_facts: list[str] | None
