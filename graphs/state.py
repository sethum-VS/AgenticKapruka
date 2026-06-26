"""LangGraph AgentState schema for the shopping assistant."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage

from lib.chat.intent_metadata import IntentMetadata

Intent = Literal["discovery", "checkout", "tracking", "general", "cart"]
DeliveryCityStatus = Literal["resolved", "ambiguous", "not_found", "missing"]
ModelTier = Literal["flash", "pro"]
AgentLoopExitReason = Literal[
    "finish",
    "ask_user",
    "max_iterations",
    "duplicate_guard",
    "tool_error",
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
    agent_tool_error: dict[str, str] | None
    model_tier: ModelTier | None
    session_id: str | None
    zep_thread_id: str | None
    currency: CurrencyCode | None
    session_budget_max: float | None
    session_budget_currency: CurrencyCode | None
    session_delivery_city_canonical: str | None
    session_delivery_date: str | None
    session_product_focus: str | None
    session_flavor_hint: str | None
    session_search_query: str | None
    session_occasion: str | None
    session_recipient_hint: str | None
    session_awaiting_delivery_date: bool | None
    session_awaiting_clarification_dimension: Literal["product", "occasion", "budget"] | None
    specificity_score: float | None
    specificity_band: Literal["proceed", "clarify", "ambiguous"] | None
    session_delivery_city_confirmed: bool | None
    session_shipment_address_raw: str | None
    delivery_city_raw: str | None
    delivery_city_canonical: str | None
    delivery_city_status: DeliveryCityStatus | None
    delivery_city_candidates: list[str] | None
    delivery_date: str | None
    delivery_context_ready: bool | None
    checkout_state: CheckoutStep | None
    response_html: str | None
    carousel_html: str | None
    assistant_message: str | None
    zep_memory_facts: list[str] | None
    last_search_products: list[dict[str, Any]] | None
    last_visible_products: list[dict[str, Any]] | None
    search_broaden_applied: bool | None
    cart_action_result: dict[str, Any] | None
