"""Routing guards for checkout, tracking, and shopping-turn preprocessing."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from graphs.state import AgentState, Intent
from lib.chat.intent_heuristics import (
    PROCEED_CHECKOUT_MESSAGE,
    classify_routing_guard,
)
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.query_preprocessor import QueryPreprocessor

logger = logging.getLogger(__name__)

# Default shopping-path intent before agent_loop refines discovery vs general.
_SHOPPING_PATH_INTENT: Intent = "discovery"

_query_preprocessor = QueryPreprocessor()


class IntentClassification(BaseModel):
    """Structured Gemini response for intent routing (legacy mocks and evals)."""

    intent: Intent


def _extract_latest_user_message(messages: list[BaseMessage]) -> str:
    """Return content of the most recent human message."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                return " ".join(parts).strip()
            return str(content)
    return ""


def _classify_routing_guard(user_message: str) -> Intent | None:
    """Return a guard intent or None when the turn should defer to agent_loop."""
    return classify_routing_guard(user_message)


def _resolve_session_budget_max(
    state: AgentState,
    intent_metadata: IntentMetadata,
) -> float | None:
    """Persist budget across turns; refresh when the user states a new cap."""
    budget = intent_metadata.get("budget_max")
    if budget is not None and budget > 0:
        return budget
    return state.get("session_budget_max")


async def analyze_intent(
    state: AgentState,
    *,
    genai_client: object | None = None,
) -> dict[str, Any]:
    """LangGraph node: guard-only routing plus query preprocessing (no LLM)."""
    _ = genai_client
    messages = state.get("messages") or []
    user_message = _extract_latest_user_message(messages)
    intent_metadata = _query_preprocessor.process(user_message)
    session_budget_max = _resolve_session_budget_max(state, intent_metadata)

    def _with_budget(payload: dict[str, Any]) -> dict[str, Any]:
        if session_budget_max is not None:
            payload["session_budget_max"] = session_budget_max
        return payload

    if not user_message.strip():
        logger.debug("analyze_intent: empty user message, defaulting to general")
        return _with_budget({"intent": "general", "intent_metadata": intent_metadata})

    if user_message.strip() == PROCEED_CHECKOUT_MESSAGE:
        logger.info("analyze_intent: proceed-to-checkout trigger from cart drawer")
        return _with_budget({"intent": "checkout", "intent_metadata": intent_metadata})

    guard_intent = _classify_routing_guard(user_message)
    if guard_intent is not None:
        logger.info("analyze_intent: guard routed message as %s", guard_intent)
        return _with_budget({"intent": guard_intent, "intent_metadata": intent_metadata})

    logger.debug(
        "analyze_intent: shopping turn — deferring discovery/general to agent_loop planner",
    )
    return _with_budget({"intent": _SHOPPING_PATH_INTENT, "intent_metadata": intent_metadata})
