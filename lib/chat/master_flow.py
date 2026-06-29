"""Flow-state supervisor: detect session/intent mismatches and patch AgentState."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, cast

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings
from graphs.state import AgentState, Intent
from lib.chat.delivery_dates import normalize_delivery_date
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.model_router import FLASH_MODEL
from lib.chat.off_topic import is_off_topic_message
from lib.chat.request_specificity import (
    ClarificationDimension,
    is_delivery_only_inquiry,
    resolve_awaiting_clarification_dimension,
    score_request_specificity,
)
from lib.chat.routing import peek_route_after_analyze_intent
from lib.checkout.chat_parser import parse_checkout_details
from lib.genai.fallback import generate_content_with_fallback
from lib.neo4j.hybrid_context import extract_budget

logger = logging.getLogger(__name__)

ActiveFlow = Literal[
    "checkout_active",
    "awaiting_delivery_date",
    "awaiting_clarification",
    "carousel_context",
    "delivery_resolution",
    "free_discovery",
]

MasterFlowDecision = Literal["proceed", "clarify", "pivot", "redirect", "checkout_exit"]

_RECIPIENT_RE = re.compile(
    r"\b(?:wife|husband|mom|mother|mum|dad|father|girlfriend|boyfriend|partner|"
    r"sister|brother|son|daughter|grandma|grandmother|grandpa|grandfather|colleague|"
    r"her|him|them)\b",
    re.I,
)

_DISCOVERY_QUESTION_RE = re.compile(
    r"\?|\b(?:what|which|show|find|recommend|browse|suggest)\b",
    re.I,
)

_CHECKOUT_EXIT_ALLOWLIST = re.compile(
    r"\b(?:"
    r"cancel(?:\s+(?:checkout|order|this))?|"
    r"stop\s+checkout|"
    r"exit\s+checkout|"
    r"leave\s+checkout|"
    r"find\s+(?:something|something\s+else)|"
    r"something\s+else|"
    r"change\s+(?:topic|subject)|"
    r"different\s+(?:gift|product)|"
    r"never\s*mind|"
    r"not\s+now|"
    r"go\s+back\s+to\s+shopping|"
    r"browse\s+(?:gifts|products)"
    r")\b",
    re.I,
)

_MASTER_FLOW_SYSTEM = """\
You are the Kapruka shopping assistant flow supervisor.

Your job is to detect when the shopper's latest message does not match the active
conversation flow, then emit structured corrections — never customer-facing prose.

Inputs describe:
- active_flow: which "chapter" the session is in
- session snapshot and awaiting flags
- planned_route: where the graph would route this turn without your help
- user_message: the latest shopper message (preserve as-is downstream)

Rules:
- Do not invent product facts or catalog results.
- Prefer decision=clarify over guessing when the mismatch is ambiguous.
- decision=pivot or redirect with context_reset=true when stale carousel/search context
  should be cleared for a fresh discovery turn.
- checkout_action=exit only when the user explicitly cancels or changes topic away
  from checkout (cancel, stop checkout, find something else, never mind).
- checkout_action=pause when the user asks an unrelated side question during checkout.
- Emit resolved_session_fields using keys downstream heuristics understand:
  session_budget_max, session_budget_currency, session_occasion, session_recipient_hint,
  session_product_focus, session_delivery_date, session_delivery_city_canonical,
  session_search_query, session_awaiting_delivery_date,
  session_awaiting_clarification_dimension.
- resolved_intent must be one of: discovery, checkout, tracking, general, cart.
- confidence reflects how certain you are; patches are applied only above threshold.
"""


class MasterFlowAlignment(BaseModel):
    """Structured Flash output for flow-state alignment."""

    decision: MasterFlowDecision
    confidence: float = Field(ge=0.0, le=1.0)
    active_flow: ActiveFlow
    mismatch_reason: str | None = None
    clarifying_question: str | None = None
    checkout_action: Literal["continue", "pause", "exit"] | None = None
    context_reset: bool = False
    resolved_intent: Intent | None = None
    resolved_session_fields: dict[str, Any] = Field(default_factory=dict)
    intent_metadata_patches: dict[str, Any] = Field(default_factory=dict)


def _extract_latest_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def _count_human_turns(messages: list[BaseMessage]) -> int:
    return sum(1 for message in messages if isinstance(message, HumanMessage))


def infer_active_flow(state: AgentState) -> ActiveFlow:
    """Derive the active conversation flow deterministically from AgentState."""
    if state.get("checkout_state"):
        return "checkout_active"
    if state.get("session_awaiting_delivery_date"):
        return "awaiting_delivery_date"
    awaiting_dim = resolve_awaiting_clarification_dimension(state)
    if awaiting_dim is not None:
        return "awaiting_clarification"
    visible = state.get("last_visible_products") or []
    if isinstance(visible, list) and visible:
        return "carousel_context"
    delivery_ready = state.get("delivery_context_ready")
    if delivery_ready is False:
        meta: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
        has_city = bool(state.get("session_delivery_city_canonical")) or bool(
            meta.get("target_city"),
        )
        has_date = bool(state.get("session_delivery_date")) or bool(meta.get("delivery_date"))
        if has_city or has_date or meta.get("requires_delivery_validation"):
            return "delivery_resolution"
    return "free_discovery"


def message_addresses_clarification_dimension(
    message: str,
    dimension: ClarificationDimension,
    state: AgentState,
) -> bool:
    """True when the message satisfies the awaited specificity dimension."""
    specificity = score_request_specificity(
        message,
        session_product_focus=state.get("session_product_focus"),
        session_occasion=state.get("session_occasion"),
        session_recipient_hint=state.get("session_recipient_hint"),
        session_budget_max=state.get("session_budget_max"),
        session_flavor_hint=state.get("session_flavor_hint"),
        intent_metadata=state.get("intent_metadata"),
    )
    threshold = 0.5 if dimension == "occasion" else 1.0
    return specificity.dimension_scores.get(dimension, 0.0) >= threshold


def is_checkout_field_answer(message: str, checkout_step: str | None) -> bool:
    """True when the message plausibly answers the active checkout step."""
    stripped = message.strip()
    if not stripped or is_off_topic_message(stripped):
        return False
    if _DISCOVERY_QUESTION_RE.search(stripped):
        return False
    details = parse_checkout_details(stripped)
    if not details:
        return False
    step = checkout_step or "cart"
    field_map: dict[str, tuple[str, ...]] = {
        "delivery_city": ("delivery_city", "delivery_address"),
        "delivery_date": ("delivery_date",),
        "recipient": ("recipient_name", "recipient_phone", "delivery_address"),
        "sender": ("sender_name",),
    }
    expected = field_map.get(step, tuple(details.keys()))
    return any(key in details for key in expected)


def message_matches_checkout_exit(message: str) -> bool:
    """Hard-coded guard rail for checkout exit on top of LLM decision."""
    return bool(_CHECKOUT_EXIT_ALLOWLIST.search(message.strip()))


def _has_stale_discovery_context(state: AgentState) -> bool:
    if state.get("session_search_query"):
        return True
    visible = state.get("last_visible_products") or []
    return isinstance(visible, list) and bool(visible)


def _long_session_drift_signals(
    state: AgentState,
    user_message: str,
    intent_metadata: IntentMetadata | dict[str, Any],
) -> bool:
    return (
        (
            extract_budget(user_message) is not None
            and not intent_metadata.get("discovery_context_reset")
        )
        or (_RECIPIENT_RE.search(user_message) is not None and _has_stale_discovery_context(state))
        or (bool(intent_metadata.get("topic_pivot")) and _has_stale_discovery_context(state))
    )


def should_invoke_master_flow(
    state: AgentState,
    *,
    settings: Settings | None = None,
) -> tuple[bool, str | None]:
    """Pure trigger gate — no LLM. Returns (invoke, reason)."""
    cfg = settings or get_settings()
    if not cfg.master_flow_enabled:
        return False, "feature_disabled"

    messages = state.get("messages") or []
    user_message = _extract_latest_user_message(messages)
    intent_metadata: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
    active_flow = infer_active_flow(state)

    if (
        state.get("session_awaiting_delivery_date")
        and normalize_delivery_date({}, user_message) is None
    ):
        return True, "awaiting_delivery_date_without_parseable_date"

    awaiting_dim = resolve_awaiting_clarification_dimension(state)
    if awaiting_dim is not None and not message_addresses_clarification_dimension(
        user_message,
        awaiting_dim,
        state,
    ):
        return True, "awaiting_clarification_dimension_unanswered"

    checkout_state = state.get("checkout_state")
    intent = state.get("intent")
    if (
        checkout_state
        and intent in ("discovery", "general")
        and not is_checkout_field_answer(user_message, checkout_state)
    ):
        return True, "checkout_active_with_discovery_intent"

    if is_delivery_only_inquiry(
        user_message,
        intent_metadata=cast(IntentMetadata | None, intent_metadata or None),
    ):
        if _has_stale_discovery_context(state):
            return True, "delivery_only_with_stale_carousel"
        planned = peek_route_after_analyze_intent(state)
        if planned == "retrieve_hybrid_context":
            return True, "delivery_only_would_reach_product_search"

    if (
        intent_metadata.get("topic_pivot") or intent_metadata.get("budgeted_gift_discovery")
    ) and _has_stale_discovery_context(state):
        return True, "topic_pivot_with_stale_carousel"

    if state.get("specificity_band") == "proceed" and active_flow == "awaiting_clarification":
        return True, "proceed_band_during_awaiting_clarification"

    long_turn_threshold = cfg.master_flow_long_session_turns
    if _count_human_turns(messages) >= long_turn_threshold and _long_session_drift_signals(
        state, user_message, intent_metadata
    ):
        return True, "long_session_drift"

    return False, None


def _context_reset_fields() -> dict[str, Any]:
    return {
        "last_visible_products": None,
        "last_search_products": None,
        "session_resolved_product": None,
        "session_search_query": None,
        "hybrid_context": {},
    }


def apply_master_flow_alignment(
    state: AgentState,
    alignment: MasterFlowAlignment,
    *,
    user_message: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Apply patches when confidence passes; return state delta."""
    cfg = settings or get_settings()
    threshold = cfg.master_flow_confidence_threshold
    updates: dict[str, Any] = {
        "master_flow_invoked": True,
        "master_flow_decision": alignment.decision,
        "active_flow": alignment.active_flow,
        "master_flow_mismatch_reason": alignment.mismatch_reason,
    }

    if alignment.confidence < threshold:
        logger.info(
            "master_flow: confidence %.2f below threshold %.2f — no patches",
            alignment.confidence,
            threshold,
        )
        return updates

    if alignment.decision == "clarify":
        question = (alignment.clarifying_question or "").strip()
        if question:
            updates["master_clarifying_question"] = question
        return updates

    if alignment.context_reset:
        updates.update(_context_reset_fields())
        meta = dict(state.get("intent_metadata") or {})
        meta["discovery_context_reset"] = True
        updates["intent_metadata"] = meta

    if alignment.resolved_intent is not None:
        updates["intent"] = alignment.resolved_intent

    for key, value in alignment.resolved_session_fields.items():
        if key in (
            "session_budget_max",
            "session_budget_currency",
            "session_occasion",
            "session_recipient_hint",
            "session_product_focus",
            "session_flavor_hint",
            "session_search_query",
            "session_delivery_date",
            "session_delivery_city_canonical",
            "session_awaiting_delivery_date",
            "session_awaiting_clarification_dimension",
        ):
            updates[key] = value

    if alignment.intent_metadata_patches:
        meta = dict(updates.get("intent_metadata") or state.get("intent_metadata") or {})
        meta.update(alignment.intent_metadata_patches)
        updates["intent_metadata"] = meta

    checkout_action = alignment.checkout_action
    if checkout_action == "pause":
        updates["checkout_paused"] = True
    elif checkout_action == "exit":
        if message_matches_checkout_exit(user_message):
            updates["checkout_state"] = None
            updates["checkout_paused"] = False
        else:
            logger.info(
                "master_flow: checkout_exit blocked — message not in exit allowlist",
            )

    return updates


def _recent_turns_summary(messages: list[BaseMessage], *, limit: int = 3) -> str:
    humans: list[str] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            text = content if isinstance(content, str) else str(content)
            humans.append(text.strip()[:120])
            if len(humans) >= limit:
                break
    humans.reverse()
    if not humans:
        return "none"
    return " | ".join(humans)


def _session_snapshot(state: AgentState) -> str:
    parts: list[str] = []
    for key in (
        "intent",
        "checkout_state",
        "checkout_paused",
        "session_budget_max",
        "session_occasion",
        "session_recipient_hint",
        "session_product_focus",
        "session_delivery_city_canonical",
        "session_delivery_date",
        "session_awaiting_delivery_date",
        "session_awaiting_clarification_dimension",
        "specificity_band",
    ):
        value = state.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    visible = state.get("last_visible_products") or []
    if isinstance(visible, list) and visible:
        parts.append(f"carousel_products={len(visible)}")
    return ", ".join(parts) if parts else "empty"


def build_master_flow_prompt(
    state: AgentState,
    *,
    active_flow: ActiveFlow,
    trigger_reason: str,
) -> str:
    """Compact user prompt for the flow supervisor."""
    user_message = _extract_latest_user_message(state.get("messages") or [])
    planned_route = peek_route_after_analyze_intent(state)
    return (
        f"active_flow: {active_flow}\n"
        f"trigger_reason: {trigger_reason}\n"
        f"planned_route: {planned_route}\n"
        f"session: {_session_snapshot(state)}\n"
        f"recent_turns: {_recent_turns_summary(state.get('messages') or [])}\n"
        f"user_message: {user_message!r}"
    )


def _parse_alignment_response(
    response: types.GenerateContentResponse,
) -> MasterFlowAlignment | None:
    parsed: MasterFlowAlignment | None = None
    if response.parsed is not None:
        try:
            if isinstance(response.parsed, MasterFlowAlignment):
                parsed = response.parsed
            else:
                parsed = MasterFlowAlignment.model_validate(response.parsed)
        except ValidationError:
            parsed = None

    if parsed is None:
        raw_text = (response.text or "").strip()
        if raw_text:
            try:
                parsed = MasterFlowAlignment.model_validate_json(raw_text)
            except Exception:
                logger.debug("master_flow: invalid JSON %r", raw_text[:200])
    return parsed


async def invoke_master_flow_llm(
    state: AgentState,
    *,
    active_flow: ActiveFlow,
    trigger_reason: str,
    genai_client: object | None = None,
) -> MasterFlowAlignment | None:
    """Call Flash with structured output; None on failure (fail-open)."""
    if not isinstance(genai_client, genai.Client):
        logger.warning("master_flow: no genai client — skipping LLM")
        return None

    user_prompt = build_master_flow_prompt(
        state,
        active_flow=active_flow,
        trigger_reason=trigger_reason,
    )

    try:
        response = generate_content_with_fallback(
            client=genai_client,
            model=FLASH_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_MASTER_FLOW_SYSTEM,
                response_mime_type="application/json",
                response_schema=MasterFlowAlignment,
                temperature=0,
            ),
        )
    except Exception:
        logger.warning("master_flow: Gemini call failed", exc_info=True)
        return None

    alignment = _parse_alignment_response(response)
    if alignment is None:
        logger.warning("master_flow: could not parse alignment response")
    return alignment
