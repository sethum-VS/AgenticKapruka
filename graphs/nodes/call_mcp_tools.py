"""Execute Kapruka MCP tools via KaprukaService based on intent and LLM tool calls."""

from __future__ import annotations

import logging
from typing import Any

from google import genai

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.checkout.tracking import extract_order_number
from lib.kapruka.product_id import extract_product_id
from lib.kapruka.service import KaprukaService
from lib.kapruka.tool_executor import SUPPORTED_TOOL_NAMES, inject_currency, invoke_tool
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.neo4j.hybrid_context import (
    build_discovery_delivery_args,
    build_discovery_search_args,
    get_discovery_occasion_hint,
    occasion_rewrite_needed,
    rewrite_search_query_with_occasion,
)

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
    hybrid_context = state.get("hybrid_context") or {}
    currency = _resolve_currency(state)

    if intent == "discovery":
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
            if delivery_args:
                calls.append({"name": CHECK_DELIVERY_TOOL, "args": delivery_args})
            return calls
        if len(user_message) >= 3:
            search_args = build_discovery_search_args(
                user_message,
                hybrid_context,
                currency=currency,
                intent_metadata=state.get("intent_metadata"),
            )
            calls = [
                {
                    "name": SEARCH_PRODUCTS_TOOL,
                    "args": search_args,
                },
            ]
            delivery_args = build_discovery_delivery_args(intent_metadata)
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
            return [{"name": TRACK_ORDER_TOOL, "args": {"order_number": order_number}}]
        return []

    return []


async def _maybe_rewrite_discovery_query(
    state: AgentState,
    args: dict[str, Any],
    *,
    genai_client: genai.Client | None = None,
) -> dict[str, Any]:
    """Apply Gemini occasion rewrite to discovery search q when graph hints require it."""
    user_query = str(args.get("q") or "").strip()
    if not user_query:
        return args

    occasion = get_discovery_occasion_hint(state.get("hybrid_context"))
    if not occasion or not occasion_rewrite_needed(user_query, occasion):
        return args

    rewritten = await rewrite_search_query_with_occasion(
        user_query,
        occasion,
        genai_client=genai_client,
    )
    return {**args, "q": rewritten}


async def call_mcp_tools(
    state: AgentState,
    *,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
    genai_client: genai.Client | None = None,
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
        if name == SEARCH_PRODUCTS_TOOL:
            args = await _maybe_rewrite_discovery_query(
                state,
                args,
                genai_client=genai_client,
            )
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
