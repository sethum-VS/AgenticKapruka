"""Shared intent preprocessing metadata schema for LangGraph state."""

from __future__ import annotations

from typing import Literal, TypedDict

Vernacular = Literal["en", "si", "tanglish"]


class IntentMetadata(TypedDict):
    """Pre-LLM signals from query preprocessing (utility vs concierge routing)."""

    is_situational: bool
    detected_vernacular: Vernacular
    requires_delivery_validation: bool
    target_city: str | None
