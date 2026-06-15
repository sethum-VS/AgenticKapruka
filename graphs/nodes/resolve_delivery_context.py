"""Pre-flight delivery city resolution before the agent loop."""

from __future__ import annotations

import logging
from typing import Any, Literal

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.chat.city_resolution import resolve_delivery_city
from lib.chat.delivery_dates import normalize_delivery_date
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.query_preprocessor import extract_target_city
from lib.kapruka.product_id import contains_product_id
from lib.kapruka.service import KaprukaService

logger = logging.getLogger(__name__)

RouteAfterResolveDeliveryContext = Literal["agent_loop", "call_mcp_tools", "generate_response"]


def route_after_resolve_delivery_context(state: AgentState) -> RouteAfterResolveDeliveryContext:
    """Route to clarify, product-ID MCP fast-path, or the agent loop."""
    clarifying = state.get("agent_clarifying_question")
    if isinstance(clarifying, str) and clarifying.strip():
        return "generate_response"

    status = state.get("delivery_city_status")
    if status in ("ambiguous", "not_found", "missing"):
        return "generate_response"

    user_message = _extract_latest_user_message(state.get("messages") or [])
    if contains_product_id(user_message):
        return "call_mcp_tools"
    return "agent_loop"


def _raw_city_from_state(state: AgentState, user_message: str) -> str | None:
    intent_metadata: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
    target_city = intent_metadata.get("target_city")
    if isinstance(target_city, str) and target_city.strip():
        return target_city.strip()
    extracted = extract_target_city(user_message)
    return extracted.strip() if extracted else None


def _needs_city_resolution(state: AgentState, user_message: str) -> bool:
    intent_metadata: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
    if intent_metadata.get("requires_delivery_validation"):
        return True
    if intent_metadata.get("target_city"):
        return True
    return extract_target_city(user_message) is not None


async def resolve_delivery_context(
    state: AgentState,
    *,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
) -> dict[str, Any]:
    """LangGraph node: canonicalize delivery city and date before catalog planning."""
    user_message = _extract_latest_user_message(state.get("messages") or [])

    if not _needs_city_resolution(state, user_message):
        return {"delivery_context_ready": True}

    if kapruka_service is None:
        msg = "kapruka_service is required for resolve_delivery_context"
        raise ValueError(msg)

    raw_city = _raw_city_from_state(state, user_message)
    rate_limit_key = client_ip or state.get("session_id") or "127.0.0.1"
    resolution = await resolve_delivery_city(kapruka_service, rate_limit_key, raw_city)

    delivery_date = normalize_delivery_date({}, user_message)
    base: dict[str, Any] = {
        "delivery_city_raw": raw_city,
        "delivery_city_status": resolution.status,
        "delivery_city_candidates": resolution.candidates,
    }
    if delivery_date is not None:
        base["delivery_date"] = delivery_date

    if resolution.status == "resolved":
        logger.info(
            "resolve_delivery_context: resolved %r -> %r",
            raw_city,
            resolution.canonical,
        )
        return {
            **base,
            "delivery_city_canonical": resolution.canonical,
            "delivery_context_ready": True,
        }

    customer_message = resolution.customer_message or "Which city should we deliver to?"
    logger.info(
        "resolve_delivery_context: %s for raw city %r — clarifying",
        resolution.status,
        raw_city,
    )
    return {
        **base,
        "delivery_context_ready": False,
        "agent_clarifying_question": customer_message,
        "agent_loop_exit_reason": "ask_user",
    }
