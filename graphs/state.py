"""LangGraph AgentState schema for the shopping assistant."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage

Intent = Literal["discovery", "checkout", "tracking", "general"]
ModelTier = Literal["flash", "pro"]
CheckoutStep = Literal[
    "cart",
    "delivery_city",
    "delivery_date",
    "recipient",
    "sender",
    "review",
]
CurrencyCode = Literal["LKR", "USD", "GBP", "AUD", "CAD", "EUR"]


class AgentState(TypedDict):
    """Shared state for the main shopping LangGraph."""

    messages: Annotated[list[BaseMessage], operator.add]
    intent: Intent | None
    hybrid_context: dict[str, Any] | None
    tool_calls: list[dict[str, Any]] | None
    tool_results: dict[str, Any] | None
    tool_call_count: int | None
    model_tier: ModelTier | None
    session_id: str | None
    zep_thread_id: str | None
    currency: CurrencyCode | None
    checkout_state: CheckoutStep | None
    response_html: str | None
