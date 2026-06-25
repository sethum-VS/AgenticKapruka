"""Shared intent preprocessing metadata schema for LangGraph state."""

from __future__ import annotations

from typing import Literal, TypedDict

Vernacular = Literal["en", "si", "tanglish"]


class IntentMetadata(TypedDict, total=False):
    """Pre-LLM signals from query preprocessing (utility vs concierge routing)."""

    is_situational: bool
    detected_vernacular: Vernacular
    requires_delivery_validation: bool
    target_city: str | None
    budget_max: float | None
    budget_currency: str | None
    redirect_kind: str | None
    is_off_topic: bool
    vernacular_score_hint: float
    topic_pivot: bool
    budgeted_gift_discovery: bool
    budget_confirmation_pending: bool
    graph_degraded: bool
