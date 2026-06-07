"""Execute Kapruka MCP tools via KaprukaService based on intent and LLM tool calls."""

from __future__ import annotations

import logging
import re
from typing import Any

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.checkout.tracking import extract_order_number
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.neo4j.hybrid_context import build_discovery_search_args

logger = logging.getLogger(__name__)

_SUPPORTED_TOOLS: frozenset[str] = frozenset(
    {SEARCH_PRODUCTS_TOOL, GET_PRODUCT_TOOL, LIST_CATEGORIES_TOOL, TRACK_ORDER_TOOL},
)

# Kapruka product IDs often embed digits (e.g. cake00ka002034, EF_PC_CHOC0V2774P00065).
_PRODUCT_ID_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*\d[A-Za-z0-9_]{2,})\b")


def _extract_product_id(user_message: str) -> str | None:
    """Return the first Kapruka-like product id token in the message, if any."""
    match = _PRODUCT_ID_RE.search(user_message)
    return match.group(1) if match else None


def _resolve_currency(state: AgentState) -> str:
    """Session currency wins; fall back to Zep hints then LKR."""
    hybrid_context = state.get("hybrid_context") or {}
    hints = hybrid_context.get("hints") or {}
    preferences = hybrid_context.get("preferences") or {}
    return state.get("currency") or hints.get("currency") or preferences.get("currency") or "LKR"


def _inject_currency_args(name: str, args: dict[str, Any], currency: str) -> dict[str, Any]:
    """Ensure price-bearing MCP tools receive the session currency."""
    if name in {SEARCH_PRODUCTS_TOOL, GET_PRODUCT_TOOL} and "currency" not in args:
        return {**args, "currency": currency}
    return args


def select_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    """Choose MCP tool invocations from explicit LLM tool_calls or routing intent."""
    explicit = state.get("tool_calls")
    if explicit:
        currency = _resolve_currency(state)
        return [
            {
                "name": call["name"],
                "args": _inject_currency_args(
                    call["name"],
                    dict(call.get("args") or {}),
                    currency,
                ),
            }
            for call in explicit
            if call.get("name") in _SUPPORTED_TOOLS
        ]

    intent = state.get("intent")
    user_message = _extract_latest_user_message(state.get("messages") or []).strip()
    hybrid_context = state.get("hybrid_context") or {}
    currency = _resolve_currency(state)

    if intent == "discovery":
        product_id = _extract_product_id(user_message)
        if product_id:
            return [
                {
                    "name": GET_PRODUCT_TOOL,
                    "args": {"product_id": product_id, "currency": currency},
                },
            ]
        if len(user_message) >= 3:
            search_args = build_discovery_search_args(
                user_message,
                hybrid_context,
                currency=currency,
            )
            return [
                {
                    "name": SEARCH_PRODUCTS_TOOL,
                    "args": search_args,
                },
            ]
        return []

    if intent == "general":
        return [{"name": LIST_CATEGORIES_TOOL, "args": {"depth": 1}}]

    if intent == "tracking":
        order_number = extract_order_number(user_message)
        if order_number:
            return [{"name": TRACK_ORDER_TOOL, "args": {"order_number": order_number}}]
        return []

    return []


def _serialize_tool_result(result: Any) -> Any:
    """Convert Pydantic tool outputs to JSON-serializable dicts for AgentState."""
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result


async def _invoke_tool(
    service: KaprukaService,
    client_ip: str,
    name: str,
    args: dict[str, Any],
) -> Any:
    """Dispatch a single supported tool call to KaprukaService."""
    if name == SEARCH_PRODUCTS_TOOL:
        return await service.search_products(client_ip, **args)
    if name == GET_PRODUCT_TOOL:
        return await service.get_product(client_ip, **args)
    if name == LIST_CATEGORIES_TOOL:
        return await service.list_categories(client_ip, **args)
    if name == TRACK_ORDER_TOOL:
        return await service.track_order(client_ip, **args)
    msg = f"Unsupported MCP tool: {name}"
    raise ValueError(msg)


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
        args = _inject_currency_args(name, dict(call.get("args") or {}), currency)
        logger.info("call_mcp_tools: invoking %s", name)
        raw = await _invoke_tool(kapruka_service, rate_limit_key, name, args)
        tool_results[name] = _serialize_tool_result(raw)
        invocations += 1

    prior_count = state.get("tool_call_count") or 0
    return {
        "tool_results": tool_results,
        "tool_call_count": prior_count + invocations,
    }
