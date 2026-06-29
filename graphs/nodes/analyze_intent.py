"""Routing guards for checkout, tracking, and shopping-turn preprocessing."""

from __future__ import annotations

import logging
import re
from typing import Any, cast

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from graphs.state import AgentState, CheckoutStep, CurrencyCode, Intent
from lib.chat.delivery_dates import (
    ambiguous_weekday_clarifying_question,
    is_ambiguous_weekday_phrase,
    normalize_delivery_date,
)
from lib.chat.intent_heuristics import (
    PROCEED_CHECKOUT_MESSAGE,
    build_guest_checkout_reply,
    classify_routing_guard,
    is_budgeted_gift_ideas_message,
    is_cart_add_trigger,
    is_guest_checkout_question,
    is_natural_budget_gift_message,
    is_order_intent_message,
    is_topic_pivot_message,
)
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.off_topic import (
    impossible_request_subject,
    is_impossible_catalog_request,
    is_off_topic_message,
    off_topic_topic,
)
from lib.chat.query_preprocessor import QueryPreprocessor
from lib.chat.request_specificity import (
    refine_specificity_with_llm,
    resolve_awaiting_clarification_dimension,
    score_request_specificity,
    should_bypass_specificity_scorer,
)
from lib.chat.support_faq import classify_support_topic, is_support_question
from lib.neo4j.hybrid_context import extract_budget
from lib.redis.cart import get_cart
from lib.redis.client import RedisClient

logger = logging.getLogger(__name__)

# Default shopping-path intent before agent_loop refines discovery vs general.
_SHOPPING_PATH_INTENT: Intent = "discovery"

_ACTIVE_CHECKOUT_STEPS: frozenset[CheckoutStep] = frozenset(
    {"delivery_city", "delivery_date", "recipient", "sender", "review"},
)

_query_preprocessor = QueryPreprocessor()

_CAKE_FOCUS = re.compile(r"\b(?:cup)?cakes?\b", re.I)
_FLOWERS_FOCUS = re.compile(
    r"\b(?:flower|flowers|rose|roses|bouquet)\b",
    re.I,
)
_FLORAL_DESIGN = re.compile(r"\b(?:floral|design|designs)\b", re.I)
_GIFT_FOCUS = re.compile(r"\b(?:gift|voucher|hamper)s?\b", re.I)
_TEA_FOCUS = re.compile(r"\b(?:tea|teas)\b", re.I)
_CHOCOLATE_FOCUS = re.compile(r"\b(?:chocolate|chocolates|cocoa|choco)\b", re.I)
_COMBO_FOCUS = re.compile(r"\b(?:combo|combopack)\b", re.I)
_OCCASION_FOCUS = re.compile(
    r"\b(?:birthday|anniversary|wedding|valentine|graduation|new baby|baby shower)\b",
    re.I,
)
_RECIPIENT_FOCUS = re.compile(
    r"\b(?:wife|husband|mom|mother|mum|dad|father|girlfriend|boyfriend|partner|"
    r"sister|brother|son|daughter|grandma|grandmother|grandpa|grandfather)\b",
    re.I,
)


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
    if _TEA_FOCUS.search(stripped):
        return "tea"
    if _GIFT_FOCUS.search(stripped):
        return "gift"
    return None


def _derive_flavor_hint(user_message: str) -> str | None:
    """Capture flavor modifiers (e.g. chocolate) even when cake wins product focus."""
    if _CHOCOLATE_FOCUS.search(user_message.strip()):
        return "chocolate"
    return None


def _resolve_session_flavor_hint(state: AgentState, user_message: str) -> str | None:
    derived = _derive_flavor_hint(user_message)
    if derived is not None:
        return derived
    prior_meta = state.get("intent_metadata") or {}
    prior = prior_meta.get("session_flavor_hint")
    if isinstance(prior, str) and prior.strip():
        return prior.strip()
    return None


def _derive_session_occasion(user_message: str) -> str | None:
    match = _OCCASION_FOCUS.search(user_message.strip())
    if match:
        return match.group(0).strip().lower()
    return None


def _derive_session_recipient_hint(user_message: str) -> str | None:
    match = _RECIPIENT_FOCUS.search(user_message.strip())
    if match:
        return match.group(0).strip().lower()
    return None


def _resolve_session_occasion(state: AgentState, user_message: str) -> str | None:
    derived = _derive_session_occasion(user_message)
    if derived is not None:
        return derived
    if is_topic_pivot_message(user_message):
        return None
    prior = state.get("session_occasion")
    if isinstance(prior, str) and prior.strip():
        return prior.strip()
    hybrid = state.get("hybrid_context") or {}
    hints = hybrid.get("hints") or {}
    occasion = hints.get("occasion")
    if isinstance(occasion, str) and occasion.strip():
        return occasion.strip().lower()
    return None


def _resolve_session_recipient_hint(state: AgentState, user_message: str) -> str | None:
    derived = _derive_session_recipient_hint(user_message)
    if derived is not None:
        return derived
    if is_topic_pivot_message(user_message):
        return None
    prior = state.get("session_recipient_hint")
    if isinstance(prior, str) and prior.strip():
        return prior.strip().lower()
    return None


def _flag_budget_confirmation_on_context_change(
    state: AgentState,
    user_message: str,
    session_budget_max: float | None,
    intent_metadata: IntentMetadata,
) -> IntentMetadata:
    """Ask once when occasion or product category changes but a session budget remains active."""
    if not isinstance(session_budget_max, (int, float)) or session_budget_max <= 0:
        return intent_metadata
    if is_topic_pivot_message(user_message):
        return intent_metadata
    # Check explicit budget in message — no need to confirm if budget is restated
    if extract_budget(user_message) is not None:
        return intent_metadata
    derived_occasion = _derive_session_occasion(user_message)
    prior_occasion = state.get("session_occasion")
    if (
        derived_occasion is not None
        and isinstance(prior_occasion, str)
        and prior_occasion.strip()
        and derived_occasion != prior_occasion.strip().lower()
    ):
        return cast(
            IntentMetadata,
            {**intent_metadata, "budget_confirmation_pending": True},
        )
    derived_focus = _derive_product_focus(user_message)
    prior_focus = state.get("session_product_focus")
    if (
        derived_focus is not None
        and isinstance(prior_focus, str)
        and prior_focus.strip()
        and derived_focus != prior_focus.strip().lower()
        # Don't ask again when switching within the same product family (e.g. chocolate ↔ cake)
        and not {derived_focus, prior_focus.strip().lower()} <= {"cake", "chocolate"}
    ):
        return cast(
            IntentMetadata,
            {**intent_metadata, "budget_confirmation_pending": True},
        )
    return intent_metadata


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
    session_occasion: str | None,
    session_recipient_hint: str | None,
    session_flavor_hint: str | None,
    delivery_date: str | None,
    session_delivery_date: str | None,
) -> dict[str, Any]:
    if session_budget_max is not None:
        payload["session_budget_max"] = session_budget_max
    if session_budget_currency is not None:
        payload["session_budget_currency"] = session_budget_currency
    if session_product_focus is not None:
        payload["session_product_focus"] = session_product_focus
    if session_occasion is not None:
        payload["session_occasion"] = session_occasion
    if session_recipient_hint is not None:
        payload["session_recipient_hint"] = session_recipient_hint
    if session_flavor_hint is not None:
        payload["session_flavor_hint"] = session_flavor_hint
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

    session_clear: dict[str, Any] = {}
    from lib.chat.product_detail import is_product_detail_turn

    if not is_cart_add_trigger(user_message) and not is_product_detail_turn(user_message):
        session_clear.update(
            {
                "last_visible_products": None,
                "last_search_products": None,
                "session_resolved_product": None,
            },
        )
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


def _clear_context_on_budget_gift_discovery(
    user_message: str,
    state: AgentState,
    intent_metadata: IntentMetadata,
) -> tuple[IntentMetadata, dict[str, Any] | None, bool, dict[str, Any]]:
    """Reset stale discovery seeds when budget + recipient starts a fresh gift search."""
    if not is_natural_budget_gift_message(user_message):
        return intent_metadata, None, False, {}

    cleared_meta = cast(
        IntentMetadata,
        {
            **intent_metadata,
            "budgeted_gift_discovery": True,
            "discovery_context_reset": True,
        },
    )
    hybrid = dict(state.get("hybrid_context") or {})
    hints = dict(hybrid.get("hints") or {})
    for key in ("occasion", "category", "exclude_categories"):
        hints.pop(key, None)
    hybrid["hints"] = hints
    hybrid["occasions"] = []

    session_clear: dict[str, Any] = {
        "last_visible_products": None,
        "last_search_products": None,
        "session_search_query": None,
        "session_occasion": None,
    }
    logger.info("analyze_intent: natural budget gift — clearing stale discovery context")
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


async def _session_cart_has_items(
    state: AgentState,
    redis_client: RedisClient | None,
) -> bool:
    """True when the session Redis cart has at least one line item."""
    session_id = state.get("session_id") or ""
    if redis_client is not None and session_id:
        rows = await get_cart(redis_client, session_id)
        if rows:
            return True
    action = state.get("cart_action_result") or {}
    cart_items = action.get("cart_items")
    return isinstance(cart_items, list) and bool(cart_items)


async def analyze_intent(
    state: AgentState,
    *,
    genai_client: object | None = None,
    redis_client: RedisClient | None = None,
) -> dict[str, Any]:
    """LangGraph node: guard-only routing plus query preprocessing (no LLM)."""
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
    budget_gift_meta, budget_hybrid_update, budget_gift_pivot, budget_session_clear = (
        _clear_context_on_budget_gift_discovery(
            user_message,
            state,
            intent_metadata,
        )
    )
    if budget_gift_pivot:
        intent_metadata = budget_gift_meta
        if budget_hybrid_update is not None:
            hybrid_context_update = budget_hybrid_update
        pivot_session_clear = {**pivot_session_clear, **budget_session_clear}
    session_product_focus = _resolve_session_product_focus(state, user_message)
    if budget_gift_pivot:
        session_product_focus = "gift"
    session_flavor_hint = _resolve_session_flavor_hint(state, user_message)
    if budget_gift_pivot:
        session_occasion = _derive_session_occasion(user_message)
        session_recipient_hint = _derive_session_recipient_hint(user_message)
    else:
        session_occasion = _resolve_session_occasion(state, user_message)
        session_recipient_hint = _resolve_session_recipient_hint(state, user_message)
    intent_metadata = _flag_budget_confirmation_on_context_change(
        state,
        user_message,
        session_budget_max,
        intent_metadata,
    )
    delivery_date, session_delivery_date = _resolve_delivery_dates(state, user_message)

    # Detect ambiguous this/next/bare weekday and ask before committing a date.
    # Remove any auto-resolved date so the session isn't poisoned with a guess.
    if is_ambiguous_weekday_phrase(user_message):
        clarification = ambiguous_weekday_clarifying_question(user_message)
        intent_metadata = cast(
            IntentMetadata,
            {
                **intent_metadata,
                "delivery_date_ambiguous": True,
                "delivery_date_clarification": clarification,
            },
        )
        delivery_date = None
        session_delivery_date = state.get("session_delivery_date")

    if is_budgeted_gift_ideas_message(user_message) or is_natural_budget_gift_message(
        user_message,
    ):
        intent_metadata = cast(
            IntentMetadata,
            {**intent_metadata, "budgeted_gift_discovery": True},
        )
    if session_flavor_hint is not None:
        intent_metadata = cast(
            IntentMetadata,
            {**intent_metadata, "session_flavor_hint": session_flavor_hint},
        )

    def _with_budget(payload: dict[str, Any]) -> dict[str, Any]:
        result = _with_session_fields(
            payload,
            session_budget_max=session_budget_max,
            session_budget_currency=session_budget_currency,
            session_product_focus=session_product_focus,
            session_occasion=session_occasion,
            session_recipient_hint=session_recipient_hint,
            session_flavor_hint=session_flavor_hint,
            delivery_date=delivery_date,
            session_delivery_date=session_delivery_date,
        )
        if topic_pivot:
            result["session_search_query"] = None
            result["session_occasion"] = None
            result["session_recipient_hint"] = None
            if session_budget_max is None:
                result["session_budget_max"] = None
                result["session_budget_currency"] = None
            result.update(pivot_session_clear)
            if hybrid_context_update is not None:
                result["hybrid_context"] = hybrid_context_update
        elif budget_gift_pivot:
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

    if is_guest_checkout_question(user_message):
        cart_has_items = await _session_cart_has_items(state, redis_client)
        logger.info(
            "analyze_intent: guest checkout info question (cart_has_items=%s)",
            cart_has_items,
        )
        intent_metadata = cast(
            IntentMetadata,
            {
                **intent_metadata,
                "guest_checkout_info": True,
                "guest_checkout_cart_has_items": cart_has_items,
            },
        )
        return _with_budget({"intent": "general", "intent_metadata": intent_metadata})

    if is_support_question(user_message):
        from lib.chat.product_detail import is_delivery_fee_question

        compound_delivery = is_delivery_fee_question(user_message)
        logger.info(
            "analyze_intent: support/policy question (%s, compound_delivery=%s)",
            classify_support_topic(user_message),
            compound_delivery,
        )
        support_meta: dict[str, Any] = {
            **intent_metadata,
            "support_topic": classify_support_topic(user_message),
            "requires_delivery_validation": compound_delivery,
        }
        if not compound_delivery:
            support_meta["target_city"] = None
        intent_metadata = cast(IntentMetadata, support_meta)
        return _with_budget({"intent": "general", "intent_metadata": intent_metadata})

    if user_message.strip() == PROCEED_CHECKOUT_MESSAGE:
        checkout_step = state.get("checkout_state")
        if checkout_step in _ACTIVE_CHECKOUT_STEPS:
            logger.info(
                "analyze_intent: suppress duplicate proceed-to-checkout at step %s",
                checkout_step,
            )
            return _with_budget(
                {
                    "intent": "general",
                    "intent_metadata": {
                        **intent_metadata,
                        "duplicate_checkout_proceed": True,
                    },
                },
            )
        logger.info("analyze_intent: proceed-to-checkout trigger from cart drawer")
        return _with_budget({"intent": "checkout", "intent_metadata": intent_metadata})

    guard_intent = _classify_routing_guard(user_message)
    if guard_intent is not None:
        logger.info("analyze_intent: guard routed message as %s", guard_intent)
        return _with_budget({"intent": guard_intent, "intent_metadata": intent_metadata})

    if is_order_intent_message(user_message) and await _session_cart_has_items(
        state,
        redis_client,
    ):
        logger.info("analyze_intent: order intent with non-empty cart -> checkout")
        return _with_budget({"intent": "checkout", "intent_metadata": intent_metadata})

    specificity_fields: dict[str, Any] = {}
    if not should_bypass_specificity_scorer(user_message, guard_intent=None):
        specificity = score_request_specificity(
            user_message,
            session_product_focus=session_product_focus,
            session_occasion=session_occasion,
            session_recipient_hint=session_recipient_hint,
            session_budget_max=session_budget_max,
            session_flavor_hint=session_flavor_hint,
            intent_metadata=intent_metadata,
        )
        if specificity.band == "ambiguous" and genai_client is not None:
            specificity = await refine_specificity_with_llm(
                user_message,
                specificity,
                genai_client=genai_client,
                session_product_focus=session_product_focus,
                session_occasion=session_occasion,
                session_recipient_hint=session_recipient_hint,
                session_budget_max=session_budget_max,
                intent_metadata=intent_metadata,
            )

        awaiting_dimension = resolve_awaiting_clarification_dimension(state)
        if awaiting_dimension is not None:
            answered_threshold = 0.5 if awaiting_dimension == "occasion" else 1.0
            if specificity.dimension_scores.get(awaiting_dimension, 0.0) >= answered_threshold:
                logger.debug(
                    "analyze_intent: specificity follow-up answered %s — proceeding",
                    awaiting_dimension,
                )
                return _with_budget(
                    {
                        "intent": _SHOPPING_PATH_INTENT,
                        "intent_metadata": intent_metadata,
                        "specificity_score": specificity.score,
                        "specificity_band": "proceed",
                        "session_awaiting_clarification_dimension": None,
                    },
                )

        if specificity.band in ("clarify", "ambiguous") and specificity.clarifying_question:
            logger.info(
                "analyze_intent: specificity gate clarify (score=%.1f, missing=%s)",
                specificity.score,
                specificity.missing_dimension,
            )
            return _with_budget(
                {
                    "intent": _SHOPPING_PATH_INTENT,
                    "intent_metadata": intent_metadata,
                    "agent_clarifying_question": specificity.clarifying_question,
                    "session_awaiting_clarification_dimension": specificity.missing_dimension,
                    "specificity_score": specificity.score,
                    "specificity_band": "clarify",
                    "tool_trace": [],
                    "tool_results": None,
                },
            )
        specificity_fields = {
            "specificity_score": specificity.score,
            "specificity_band": specificity.band,
        }
        if resolve_awaiting_clarification_dimension(state):
            logger.debug(
                "analyze_intent: specificity gate proceed (score=%.1f) — clearing await flag",
                specificity.score,
            )
            specificity_fields["session_awaiting_clarification_dimension"] = None
        else:
            logger.debug(
                "analyze_intent: specificity gate proceed (score=%.1f)",
                specificity.score,
            )

    logger.debug(
        "analyze_intent: shopping turn — deferring discovery/general to agent_loop planner",
    )
    return _with_budget(
        {
            "intent": _SHOPPING_PATH_INTENT,
            "intent_metadata": intent_metadata,
            **specificity_fields,
        },
    )
