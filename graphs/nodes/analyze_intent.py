"""Routing guards for checkout, tracking, and shopping-turn preprocessing."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from graphs.state import AgentState, Intent
from lib.chat.intent_heuristics import (
    PROCEED_CHECKOUT_MESSAGE,
    is_checkout_trigger,
    is_proceed_checkout_message,
    is_tracking_guard,
)
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
    if is_proceed_checkout_message(user_message):
        return "checkout"
    if is_tracking_guard(user_message):
        return "tracking"
    if is_checkout_trigger(user_message):
        return "checkout"
    return None


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

    if not user_message.strip():
        logger.debug("analyze_intent: empty user message, defaulting to general")
        return {"intent": "general", "intent_metadata": intent_metadata}

    if user_message.strip() == PROCEED_CHECKOUT_MESSAGE:
        logger.info("analyze_intent: proceed-to-checkout trigger from cart drawer")
        return {"intent": "checkout", "intent_metadata": intent_metadata}

    guard_intent = _classify_routing_guard(user_message)
    if guard_intent is not None:
        logger.info("analyze_intent: guard routed message as %s", guard_intent)
        return {"intent": guard_intent, "intent_metadata": intent_metadata}

    logger.debug(
        "analyze_intent: shopping turn — deferring discovery/general to agent_loop planner",
    )
    return {"intent": _SHOPPING_PATH_INTENT, "intent_metadata": intent_metadata}
