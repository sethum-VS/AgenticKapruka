"""Execute Kapruka MCP tools via KaprukaService based on intent and LLM tool calls."""

from __future__ import annotations

import logging
from typing import Any

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.checkout.tracking import classify_order_reference, extract_order_number
from lib.kapruka.product_id import extract_product_id
from lib.kapruka.service import KaprukaService
from lib.kapruka.tool_executor import SUPPORTED_TOOL_NAMES, inject_currency, invoke_tool
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.neo4j.hybrid_context import build_discovery_delivery_args

logger = logging.getLogger(__name__)


def _resolve_currency(state: AgentState) -> str:
    """Session currency wins; fall back to Zep hints then LKR."""
    hybrid_context = state.get("hybrid_context") or {}
    hints = hybrid_context.get("hints") or {}
    preferences = hybrid_context.get("preferences") or {}
    return state.get("currency") or hints.get("currency") or preferences.get("currency") or "LKR"


def select_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    """Choose MCP tool invocations from explicit LLM tool_calls or routing intent."""
    explicit = state.get("tool_calls")
    if explicit:
        currency = _resolve_currency(state)
        return [
            {
                "name": call["name"],
                "args": inject_currency(
                    call["name"],
                    dict(call.get("args") or {}),
                    currency,
                ),
            }
            for call in explicit
            if call.get("name") in SUPPORTED_TOOL_NAMES
        ]

    intent = state.get("intent")
    user_message = _extract_latest_user_message(state.get("messages") or []).strip()
    currency = _resolve_currency(state)

    if intent == "discovery":
        # Discovery catalog turns route through agent_loop (PRD-107/108). Only the
        # product-ID fast-path and explicit tool_calls reach this node.
        intent_metadata = state.get("intent_metadata")
        product_id = extract_product_id(user_message)
        if product_id:
            calls: list[dict[str, Any]] = [
                {
                    "name": GET_PRODUCT_TOOL,
                    "args": {"product_id": product_id, "currency": currency},
                },
            ]
            delivery_args = build_discovery_delivery_args(intent_metadata)
            canonical_city = state.get("delivery_city_canonical")
            if isinstance(canonical_city, str) and canonical_city.strip():
                delivery_args["city"] = canonical_city.strip()
            elif delivery_args.get("city") and intent_metadata:
                target = intent_metadata.get("target_city")
                if isinstance(target, str) and target.strip():
                    delivery_args["city"] = target.strip()
            state_date = state.get("delivery_date")
            if isinstance(state_date, str) and state_date.strip():
                delivery_args["delivery_date"] = state_date.strip()
            if delivery_args:
                calls.append({"name": CHECK_DELIVERY_TOOL, "args": delivery_args})
            return calls
        return []

    if intent == "general":
        product_id = extract_product_id(user_message)
        if product_id:
            return [
                {
                    "name": GET_PRODUCT_TOOL,
                    "args": {"product_id": product_id, "currency": currency},
                },
            ]
        return [{"name": LIST_CATEGORIES_TOOL, "args": {"depth": 1}}]

    if intent == "tracking":
        order_number = extract_order_number(user_message)
        if order_number:
            if classify_order_reference(order_number) == "ka_legacy":
                return []
            return [{"name": TRACK_ORDER_TOOL, "args": {"order_number": order_number}}]
        return []

    return []


async def call_mcp_tools(
    state: AgentState,
    *,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
) -> dict[str, Any]:
    """LangGraph node: invoke Kapruka MCP tools and accumulate tool_results."""
    if kapruka_service is None:
        msg = "kapruka_service is required for call_mcp_tools"
        raise ValueError(msg)

    rate_limit_key = client_ip or state.get("session_id") or "127.0.0.1"
    selected = select_tool_calls(state)
    if not selected:
        logger.debug("call_mcp_tools: no tools selected for intent=%s", state.get("intent"))
        return {"tool_results": {}}

    tool_results: dict[str, Any] = {}
    invocations = 0

    currency = _resolve_currency(state)

    for call in selected:
        name = call["name"]
        args = inject_currency(name, dict(call.get("args") or {}), currency)
        logger.info("call_mcp_tools: invoking %s", name)
        tool_results[name] = await invoke_tool(
            name,
            args,
            kapruka_service=kapruka_service,
            client_ip=rate_limit_key,
            currency=currency,
        )
        invocations += 1

    prior_count = state.get("tool_call_count") or 0
    return {
        "tool_results": tool_results,
        "tool_call_count": prior_count + invocations,
    }
