"""Pre-flight delivery city resolution before the agent loop."""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from google import genai

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, ToolInvocation
from lib.chat.address_resolution import resolve_shipment_address
from lib.chat.city_resolution import _is_bare_colombo, resolve_delivery_city
from lib.chat.delivery_dates import is_delivery_date_only_message, normalize_delivery_date
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.query_preprocessor import (
    _has_delivery_intent,
    _has_perishable_gift_intent,
    extract_target_city,
)
from lib.chat.request_specificity import is_delivery_only_inquiry
from lib.kapruka.product_id import contains_product_id
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL

logger = logging.getLogger(__name__)

RouteAfterResolveDeliveryContext = Literal["agent_loop", "call_mcp_tools", "generate_response"]


def route_after_resolve_delivery_context(state: AgentState) -> RouteAfterResolveDeliveryContext:
    """Route to product-ID MCP fast-path or the agent loop (clarify+search)."""
    intent_metadata: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
    if intent_metadata.get("support_topic"):
        return "generate_response"
    user_message = _extract_latest_user_message(state.get("messages") or [])
    if contains_product_id(user_message):
        return "call_mcp_tools"
    if is_delivery_only_inquiry(
        user_message,
        intent_metadata=cast(IntentMetadata | None, intent_metadata or None),
    ):
        return "generate_response"
    return "agent_loop"


def _resolve_session_delivery_city(state: AgentState, user_message: str) -> str | None:
    """Prefer an explicit city this turn; otherwise reuse the session canonical city."""
    intent_metadata: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
    target_city = intent_metadata.get("target_city")
    if isinstance(target_city, str) and target_city.strip():
        return target_city.strip()
    extracted = extract_target_city(user_message)
    if extracted:
        return extracted.strip()
    session_city = state.get("session_delivery_city_canonical")
    if isinstance(session_city, str) and session_city.strip():
        return session_city.strip()
    return None


def _should_soft_colombo_zone_nudge(state: AgentState, user_message: str) -> bool:
    """Gift discovery with bare 'Colombo' gets a gentle zone nudge, not a hard stop."""
    if _has_delivery_intent(user_message):
        return False
    raw_city = extract_target_city(user_message)
    if not isinstance(raw_city, str) or not _is_bare_colombo(raw_city):
        return False
    return _has_perishable_gift_intent(user_message)


def _needs_city_resolution(state: AgentState, user_message: str) -> bool:
    intent_metadata: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
    if intent_metadata.get("requires_delivery_validation"):
        return True
    if intent_metadata.get("target_city"):
        return True
    if extract_target_city(user_message) is not None:
        return True
    if state.get("session_awaiting_delivery_date"):
        return True
    session_city = state.get("session_delivery_city_canonical")
    if (
        isinstance(session_city, str)
        and bool(session_city.strip())
        and is_delivery_date_only_message(user_message)
    ):
        return True
    return (
        isinstance(session_city, str)
        and bool(session_city.strip())
        and _has_delivery_intent(user_message)
    )


def _city_ready_for_delivery_preflight(
    state: AgentState,
    *,
    city_confirmed: bool = False,
) -> bool:
    """True when canonical city is known enough for kapruka_check_delivery preflight."""
    if city_confirmed or state.get("session_delivery_city_confirmed"):
        return True
    status = state.get("delivery_city_status")
    if status == "resolved":
        return True
    canonical = state.get("session_delivery_city_canonical") or state.get(
        "delivery_city_canonical",
    )
    return (
        isinstance(canonical, str) and bool(canonical.strip()) and status not in ("ambiguous", None)
    )


def _should_run_delivery_preflight(
    state: AgentState,
    user_message: str,
    intent_metadata: IntentMetadata | dict[str, Any],
    *,
    delivery_date: str | None,
    city_confirmed: bool = False,
) -> bool:
    """City-only kapruka_check_delivery before the agent loop asks for a date."""
    if delivery_date is not None:
        return False
    if not _city_ready_for_delivery_preflight(state, city_confirmed=city_confirmed):
        return False
    if intent_metadata.get("requires_delivery_validation"):
        return True
    session_city = state.get("session_delivery_city_canonical")
    return (
        isinstance(session_city, str)
        and bool(session_city.strip())
        and _has_delivery_intent(user_message)
    )


def _should_run_dated_delivery_preflight(
    state: AgentState,
    user_message: str,
    intent_metadata: IntentMetadata | dict[str, Any],
    *,
    delivery_date: str | None,
    city_confirmed: bool = False,
) -> bool:
    """Dated kapruka_check_delivery when canonical city and delivery date are both known."""
    if delivery_date is None:
        return False
    if not _city_ready_for_delivery_preflight(state, city_confirmed=city_confirmed):
        return False
    if intent_metadata.get("requires_delivery_validation"):
        return True
    session_city = state.get("session_delivery_city_canonical")
    if (
        isinstance(session_city, str)
        and bool(session_city.strip())
        and is_delivery_date_only_message(user_message)
    ):
        return True
    return (
        isinstance(session_city, str)
        and bool(session_city.strip())
        and _has_delivery_intent(user_message)
    )


def _resolve_delivery_product_id(state: AgentState) -> str | None:
    """Pick a catalog product id for perishable-aware delivery checks."""
    for key in ("last_visible_products", "last_search_products"):
        products = state.get(key)
        if not isinstance(products, list) or not products:
            continue
        first = products[0]
        if isinstance(first, dict):
            product_id = first.get("id")
            if isinstance(product_id, str) and product_id.strip():
                return product_id.strip()
    return None


async def _preflight_check_delivery(
    *,
    kapruka_service: KaprukaService,
    client_ip: str,
    city: str,
    delivery_date: str | None = None,
    product_id: str | None = None,
) -> ToolInvocation:
    """Run check_delivery and return a tool_trace entry."""
    if delivery_date:
        output = await kapruka_service.check_delivery(
            client_ip,
            city=city,
            delivery_date=delivery_date,
            product_id=product_id,
        )
        args: dict[str, str] = {"city": city, "delivery_date": delivery_date}
    else:
        output = await kapruka_service.check_delivery(
            client_ip,
            city=city,
            product_id=product_id,
        )
        args = {"city": city}
    if product_id:
        args["product_id"] = product_id
    return {
        "name": CHECK_DELIVERY_TOOL,
        "args": args,
        "result": output.model_dump(),
    }


async def resolve_delivery_context(
    state: AgentState,
    *,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
    genai_client: genai.Client | None = None,
) -> dict[str, Any]:
    """LangGraph node: canonicalize delivery city and date before catalog planning."""
    user_message = _extract_latest_user_message(state.get("messages") or [])

    if not _needs_city_resolution(state, user_message):
        return {"delivery_context_ready": True}

    if kapruka_service is None:
        msg = "kapruka_service is required for resolve_delivery_context"
        raise ValueError(msg)

    rate_limit_key = client_ip or state.get("session_id") or "127.0.0.1"

    address_updates: dict[str, Any] = {}
    if genai_client is not None:
        address_updates = await resolve_shipment_address(
            state,
            kapruka_service=kapruka_service,
            client_ip=rate_limit_key,
            genai_client=genai_client,
        )
        if address_updates.get("agent_clarifying_question"):
            return address_updates

    raw_city = _resolve_session_delivery_city(state, user_message)
    if not raw_city and address_updates.get("delivery_city_raw"):
        raw_city = str(address_updates.get("delivery_city_raw"))

    if address_updates.get("delivery_city_canonical"):
        resolution_status = address_updates.get("delivery_city_status", "resolved")
        delivery_date = normalize_delivery_date({}, user_message)
        base: dict[str, Any] = {
            **address_updates,
            "delivery_context_ready": address_updates.get("delivery_context_ready", True),
        }
        if delivery_date is not None:
            base["delivery_date"] = delivery_date
            base["session_delivery_date"] = delivery_date
        if resolution_status == "resolved" and base.get("session_delivery_city_confirmed"):
            intent_metadata: IntentMetadata | dict[str, Any] = state.get("intent_metadata") or {}
            canonical = base.get("delivery_city_canonical")
            if (
                canonical
                and delivery_date
                and _should_run_dated_delivery_preflight(
                    state,
                    user_message,
                    intent_metadata,
                    delivery_date=delivery_date,
                    city_confirmed=bool(base.get("session_delivery_city_confirmed")),
                )
            ):
                preflight = await _preflight_check_delivery(
                    kapruka_service=kapruka_service,
                    client_ip=rate_limit_key,
                    city=str(canonical),
                    delivery_date=delivery_date,
                    product_id=_resolve_delivery_product_id(state),
                )
                base["tool_trace"] = [preflight]
            elif canonical and _should_run_delivery_preflight(
                state,
                user_message,
                intent_metadata,
                delivery_date=delivery_date,
                city_confirmed=bool(base.get("session_delivery_city_confirmed")),
            ):
                preflight = await _preflight_check_delivery(
                    kapruka_service=kapruka_service,
                    client_ip=rate_limit_key,
                    city=str(canonical),
                    product_id=_resolve_delivery_product_id(state),
                )
                base["tool_trace"] = [preflight]
            if base.get("tool_trace"):
                preflight_result = (base["tool_trace"][0]).get("result")
                if isinstance(preflight_result, dict) and not preflight_result.get("available"):
                    reason = preflight_result.get("reason")
                    customer_message = (
                        str(reason).strip()
                        if isinstance(reason, str) and reason.strip()
                        else f"Kapruka cannot deliver to {canonical}."
                    )
                    return {
                        **base,
                        "delivery_context_ready": False,
                        "agent_clarifying_question": customer_message,
                        "agent_loop_exit_reason": "ask_user",
                    }
        return base

    resolution = await resolve_delivery_city(kapruka_service, rate_limit_key, raw_city)

    delivery_date = normalize_delivery_date({}, user_message)
    resolved_base: dict[str, Any] = {
        **address_updates,
        "delivery_city_raw": raw_city or address_updates.get("delivery_city_raw"),
        "delivery_city_status": resolution.status,
        "delivery_city_candidates": resolution.candidates,
    }
    if delivery_date is not None:
        resolved_base["delivery_date"] = delivery_date
        resolved_base["session_delivery_date"] = delivery_date

    if resolution.status == "resolved":
        logger.info(
            "resolve_delivery_context: resolved %r -> %r",
            raw_city,
            resolution.canonical,
        )
        resolved: dict[str, Any] = {
            **resolved_base,
            "delivery_city_canonical": resolution.canonical,
            "session_delivery_city_canonical": resolution.canonical,
            "session_delivery_city_confirmed": True,
            "delivery_context_ready": True,
        }
        intent_metadata = state.get("intent_metadata") or {}
        canonical = resolution.canonical
        if (
            canonical
            and delivery_date
            and _should_run_dated_delivery_preflight(
                state,
                user_message,
                intent_metadata,
                delivery_date=delivery_date,
                city_confirmed=True,
            )
        ):
            preflight = await _preflight_check_delivery(
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                city=canonical,
                delivery_date=delivery_date,
                product_id=_resolve_delivery_product_id(state),
            )
            resolved["tool_trace"] = [preflight]
        elif canonical and _should_run_delivery_preflight(
            state,
            user_message,
            intent_metadata,
            delivery_date=delivery_date,
            city_confirmed=True,
        ):
            preflight = await _preflight_check_delivery(
                kapruka_service=kapruka_service,
                client_ip=rate_limit_key,
                city=canonical,
                product_id=_resolve_delivery_product_id(state),
            )
            resolved["tool_trace"] = [preflight]
        if resolved.get("tool_trace"):
            preflight_result = resolved["tool_trace"][0].get("result")
            if isinstance(preflight_result, dict) and not preflight_result.get("available"):
                reason = preflight_result.get("reason")
                customer_message = (
                    str(reason).strip()
                    if isinstance(reason, str) and reason.strip()
                    else f"Kapruka cannot deliver to {canonical}."
                )
                return {
                    **resolved,
                    "delivery_context_ready": False,
                    "agent_clarifying_question": customer_message,
                    "agent_loop_exit_reason": "ask_user",
                }
        return resolved

    customer_message = resolution.customer_message or "Which city should we deliver to?"
    logger.info(
        "resolve_delivery_context: %s for raw city %r — clarifying",
        resolution.status,
        raw_city,
    )
    if resolution.status == "ambiguous" and _should_soft_colombo_zone_nudge(state, user_message):
        logger.info(
            "resolve_delivery_context: soft Colombo zone nudge for gift discovery %r",
            raw_city,
        )
        return {
            **resolved_base,
            "delivery_city_raw": raw_city,
            "delivery_city_status": "ambiguous",
            "delivery_city_candidates": resolution.candidates,
            "delivery_context_ready": True,
            "agent_clarifying_question": customer_message,
        }
    return {
        **resolved_base,
        "delivery_context_ready": False,
        "agent_clarifying_question": customer_message,
        "agent_loop_exit_reason": "ask_user",
    }
