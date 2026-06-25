"""Routing guards for checkout, tracking, and shopping-turn preprocessing."""

from __future__ import annotations

import logging
import re
from typing import Any, cast

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from graphs.state import AgentState, CurrencyCode, Intent
from lib.chat.delivery_dates import normalize_delivery_date
from lib.chat.intent_heuristics import (
    GIFT_PREFERENCES_QUESTION,
    PROCEED_CHECKOUT_MESSAGE,
    classify_routing_guard,
    is_topic_pivot_message,
    is_vague_gift_intent,
)
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.off_topic import (
    impossible_request_subject,
    is_impossible_catalog_request,
    is_off_topic_message,
    off_topic_topic,
)
from lib.chat.query_preprocessor import QueryPreprocessor
from lib.neo4j.hybrid_context import extract_budget

logger = logging.getLogger(__name__)

# Default shopping-path intent before agent_loop refines discovery vs general.
_SHOPPING_PATH_INTENT: Intent = "discovery"

_query_preprocessor = QueryPreprocessor()

_CAKE_FOCUS = re.compile(r"\b(?:cup)?cakes?\b", re.I)
_FLOWERS_FOCUS = re.compile(
    r"\b(?:flower|flowers|rose|roses|bouquet)\b",
    re.I,
)
_FLORAL_DESIGN = re.compile(r"\b(?:floral|design|designs)\b", re.I)
_GIFT_FOCUS = re.compile(r"\b(?:gift|voucher|hamper)s?\b", re.I)
_CHOCOLATE_FOCUS = re.compile(r"\b(?:chocolate|chocolates|cocoa|choco)\b", re.I)
_COMBO_FOCUS = re.compile(r"\b(?:combo|combopack)\b", re.I)


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


def _resolve_session_budget(
    state: AgentState,
    user_message: str,
    intent_metadata: IntentMetadata,
) -> tuple[float | None, CurrencyCode | None]:
    """Persist budget across turns; message currency wins when explicit."""
    cap = extract_budget(user_message)
    if cap is not None:
        valid_currencies = ("LKR", "USD", "GBP", "AUD", "CAD", "EUR")
        currency = cap.currency if cap.currency in valid_currencies else "LKR"
        return cap.amount, currency  # type: ignore[return-value]

    budget = intent_metadata.get("budget_max")
    if budget is not None and budget > 0:
        session_currency = state.get("session_budget_currency")
        if session_currency in ("LKR", "USD", "GBP", "AUD", "CAD", "EUR"):
            return budget, session_currency
        return budget, None

    session_max = state.get("session_budget_max")
    if isinstance(session_max, (int, float)) and session_max > 0:
        session_currency = state.get("session_budget_currency")
        if session_currency in ("LKR", "USD", "GBP", "AUD", "CAD", "EUR"):
            return float(session_max), session_currency
        return float(session_max), None

    return None, state.get("session_budget_currency")


def _derive_product_focus(user_message: str) -> str | None:
    """Infer shopping focus from explicit product mentions in the current turn."""
    stripped = user_message.strip()
    if not stripped:
        return None
    if _COMBO_FOCUS.search(stripped):
        return "combo"
    if _CAKE_FOCUS.search(stripped):
        return "cake"
    if _FLOWERS_FOCUS.search(stripped):
        return "flowers"
    if _CHOCOLATE_FOCUS.search(stripped):
        return "chocolate"
    if _GIFT_FOCUS.search(stripped):
        return "gift"
    return None


def _resolve_session_product_focus(state: AgentState, user_message: str) -> str | None:
    """Persist product focus across turns; refresh on explicit mentions."""
    prior = state.get("session_product_focus")
    derived = _derive_product_focus(user_message)
    if derived is None:
        return prior
    if (
        prior == "cake"
        and derived == "flowers"
        and _FLORAL_DESIGN.search(user_message)
        and not _FLOWERS_FOCUS.search(user_message)
    ):
        return prior
    return derived


def _resolve_delivery_dates(
    state: AgentState,
    user_message: str,
) -> tuple[str | None, str | None]:
    """Return (delivery_date for this turn, persisted session_delivery_date)."""
    parsed = normalize_delivery_date({}, user_message)
    if parsed is not None:
        return parsed, parsed
    session_date = state.get("session_delivery_date")
    if isinstance(session_date, str) and session_date.strip():
        return session_date.strip(), session_date.strip()
    return None, state.get("session_delivery_date")


def _with_session_fields(
    payload: dict[str, Any],
    *,
    session_budget_max: float | None,
    session_budget_currency: CurrencyCode | None,
    session_product_focus: str | None,
    delivery_date: str | None,
    session_delivery_date: str | None,
) -> dict[str, Any]:
    if session_budget_max is not None:
        payload["session_budget_max"] = session_budget_max
    if session_budget_currency is not None:
        payload["session_budget_currency"] = session_budget_currency
    if session_product_focus is not None:
        payload["session_product_focus"] = session_product_focus
    if delivery_date is not None:
        payload["delivery_date"] = delivery_date
    if session_delivery_date is not None:
        payload["session_delivery_date"] = session_delivery_date
    return payload


def _clear_budget_on_pivot(
    user_message: str,
    session_budget_max: float | None,
    session_budget_currency: CurrencyCode | None,
    intent_metadata: IntentMetadata,
) -> tuple[float | None, CurrencyCode | None, IntentMetadata]:
    """Drop sticky budget when the customer pivots to a new product topic."""
    if not is_topic_pivot_message(user_message):
        return session_budget_max, session_budget_currency, intent_metadata
    if extract_budget(user_message) is not None:
        return session_budget_max, session_budget_currency, intent_metadata
    cleared = cast(IntentMetadata, {**intent_metadata, "budget_max": None, "budget_currency": None})
    logger.info("analyze_intent: topic pivot — clearing session budget")
    return None, None, cleared


def _clear_context_on_pivot(
    user_message: str,
    state: AgentState,
    intent_metadata: IntentMetadata,
) -> tuple[IntentMetadata, dict[str, Any] | None, bool, dict[str, Any]]:
    """Drop sticky occasion/search seeds when the customer pivots without a new occasion."""
    if not is_topic_pivot_message(user_message):
        return intent_metadata, None, False, {}

    cleared_meta = cast(IntentMetadata, {**intent_metadata, "topic_pivot": True})
    hybrid = dict(state.get("hybrid_context") or {})
    hints = dict(hybrid.get("hints") or {})
    for key in ("occasion", "category", "exclude_categories"):
        hints.pop(key, None)
    hybrid["hints"] = hints
    hybrid["occasions"] = []

    session_clear: dict[str, Any] = {
        "last_visible_products": None,
        "last_search_products": None,
    }
    from lib.chat.delivery_dates import normalize_delivery_date
    from lib.chat.query_preprocessor import extract_target_city

    city_missing = extract_target_city(user_message) is None
    date_missing = normalize_delivery_date({}, user_message) is None
    if city_missing and date_missing:
        session_clear.update(
            {
                "session_delivery_city_canonical": None,
                "session_delivery_date": None,
                "session_awaiting_delivery_date": False,
                "delivery_city_canonical": None,
                "delivery_date": None,
            },
        )

    logger.info("analyze_intent: topic pivot — clearing hybrid occasion/search seeds")
    return cleared_meta, hybrid, True, session_clear


def _off_topic_metadata(
    intent_metadata: IntentMetadata,
    *,
    redirect_kind: str,
) -> IntentMetadata:
    return cast(
        IntentMetadata,
        {
            **intent_metadata,
            "redirect_kind": redirect_kind,
            "is_off_topic": True,
            "requires_delivery_validation": False,
            "target_city": None,
        },
    )


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
    session_budget_max, session_budget_currency = _resolve_session_budget(
        state,
        user_message,
        intent_metadata,
    )
    session_budget_max, session_budget_currency, intent_metadata = _clear_budget_on_pivot(
        user_message,
        session_budget_max,
        session_budget_currency,
        intent_metadata,
    )
    intent_metadata, hybrid_context_update, topic_pivot, pivot_session_clear = (
        _clear_context_on_pivot(
        user_message,
        state,
        intent_metadata,
        )
    )
    session_product_focus = _resolve_session_product_focus(state, user_message)
    delivery_date, session_delivery_date = _resolve_delivery_dates(state, user_message)

    def _with_budget(payload: dict[str, Any]) -> dict[str, Any]:
        result = _with_session_fields(
            payload,
            session_budget_max=session_budget_max,
            session_budget_currency=session_budget_currency,
            session_product_focus=session_product_focus,
            delivery_date=delivery_date,
            session_delivery_date=session_delivery_date,
        )
        if topic_pivot:
            result["session_search_query"] = None
            result.update(pivot_session_clear)
            if hybrid_context_update is not None:
                result["hybrid_context"] = hybrid_context_update
        return result

    if not user_message.strip():
        logger.debug("analyze_intent: empty user message, defaulting to general")
        return _with_budget({"intent": "general", "intent_metadata": intent_metadata})

    if is_impossible_catalog_request(user_message):
        logger.info(
            "analyze_intent: impossible catalog request (%s)",
            impossible_request_subject(user_message),
        )
        return _with_budget(
            {
                "intent": "general",
                "intent_metadata": _off_topic_metadata(
                    intent_metadata,
                    redirect_kind="impossible_product",
                ),
            },
        )

    if is_off_topic_message(user_message):
        logger.info("analyze_intent: off-topic message (%s)", off_topic_topic(user_message))
        return _with_budget(
            {
                "intent": "general",
                "intent_metadata": _off_topic_metadata(
                    intent_metadata,
                    redirect_kind="off_topic",
                ),
            },
        )

    if user_message.strip() == PROCEED_CHECKOUT_MESSAGE:
        logger.info("analyze_intent: proceed-to-checkout trigger from cart drawer")
        return _with_budget({"intent": "checkout", "intent_metadata": intent_metadata})

    guard_intent = _classify_routing_guard(user_message)
    if guard_intent is not None:
        logger.info("analyze_intent: guard routed message as %s", guard_intent)
        return _with_budget({"intent": guard_intent, "intent_metadata": intent_metadata})

    if state.get("session_awaiting_gift_preferences"):
        logger.debug("analyze_intent: gift preferences follow-up — proceeding to search")
        return _with_budget(
            {
                "intent": _SHOPPING_PATH_INTENT,
                "intent_metadata": intent_metadata,
                "session_awaiting_gift_preferences": False,
            },
        )

    if is_vague_gift_intent(user_message):
        logger.info("analyze_intent: vague gift query — asking for preferences")
        question = GIFT_PREFERENCES_QUESTION
        if intent_metadata.get("is_situational"):
            question = (
                "I'm sorry to hear you're going through this. "
                f"{question}"
            )
        return _with_budget(
            {
                "intent": _SHOPPING_PATH_INTENT,
                "intent_metadata": intent_metadata,
                "agent_clarifying_question": question,
                "session_awaiting_gift_preferences": True,
            },
        )

    logger.debug(
        "analyze_intent: shopping turn — deferring discovery/general to agent_loop planner",
    )
    return _with_budget({"intent": _SHOPPING_PATH_INTENT, "intent_metadata": intent_metadata})
