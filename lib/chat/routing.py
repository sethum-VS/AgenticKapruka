"""Shared post-intent routing for the shopping LangGraph."""

from __future__ import annotations

import logging
from typing import Literal, cast

from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, Intent
from lib.chat.intent_metadata import IntentMetadata
from lib.chat.request_specificity import is_delivery_only_inquiry
from lib.debug.trace import trace_route_decision
from lib.kapruka.product_id import contains_product_id

logger = logging.getLogger(__name__)

RouteAfterAnalyzeIntent = Literal[
    "retrieve_hybrid_context",
    "call_mcp_tools",
    "run_checkout_graph",
    "resolve_cart_product",
    "resolve_delivery_context",
    "generate_response",
]

_INTENTS_SKIP_HYBRID_CONTEXT: frozenset[Intent] = frozenset({"tracking"})


def peek_route_after_analyze_intent(state: AgentState) -> RouteAfterAnalyzeIntent:
    """Planned route after analyze_intent (and master_flow) without side effects."""
    return route_after_analyze_intent(state)


def route_after_analyze_intent(state: AgentState) -> RouteAfterAnalyzeIntent:
    """Conditional edge after analyze_intent: checkout sub-graph or HybridRAG skip."""
    intent_metadata: IntentMetadata | dict[str, object] = state.get("intent_metadata") or {}
    if intent_metadata.get("is_off_topic"):
        trace_route_decision(
            from_node="analyze_intent",
            target="generate_response",
            intent=state.get("intent"),
            reason="off-topic or impossible catalog redirect",
        )
        return "generate_response"
    user_message = _extract_latest_user_message(state.get("messages") or [])
    if is_delivery_only_inquiry(
        user_message,
        intent_metadata=cast(IntentMetadata | None, intent_metadata or None),
    ):
        trace_route_decision(
            from_node="analyze_intent",
            target="resolve_delivery_context",
            intent=state.get("intent"),
            reason="delivery-only inquiry with city and date",
        )
        return "resolve_delivery_context"
    clarifying = state.get("agent_clarifying_question")
    if (
        state.get("specificity_band") == "clarify"
        and isinstance(clarifying, str)
        and clarifying.strip()
    ):
        trace_route_decision(
            from_node="analyze_intent",
            target="generate_response",
            intent=state.get("intent"),
            reason="specificity gate — clarify without catalog search",
        )
        return "generate_response"
    if intent_metadata.get("support_topic"):
        if intent_metadata.get("requires_delivery_validation"):
            trace_route_decision(
                from_node="analyze_intent",
                target="resolve_delivery_context",
                intent=state.get("intent"),
                reason="support FAQ with delivery fee question",
            )
            return "resolve_delivery_context"
        trace_route_decision(
            from_node="analyze_intent",
            target="generate_response",
            intent=state.get("intent"),
            reason="support or policy FAQ",
        )
        return "generate_response"

    intent_metadata = state.get("intent_metadata") or {}
    if intent_metadata.get("duplicate_checkout_proceed"):
        trace_route_decision(
            from_node="analyze_intent",
            target="generate_response",
            intent=state.get("intent"),
            reason="duplicate proceed-to-checkout suppressed",
        )
        return "generate_response"

    intent = state.get("intent")
    if intent == "checkout":
        logger.debug("route_after_analyze_intent: routing to checkout sub-graph")
        trace_route_decision(
            from_node="analyze_intent",
            target="run_checkout_graph",
            intent=intent,
            reason="checkout intent",
        )
        return "run_checkout_graph"
    if intent == "cart":
        logger.debug("route_after_analyze_intent: routing to cart transaction path")
        trace_route_decision(
            from_node="analyze_intent",
            target="resolve_cart_product",
            intent=intent,
            reason="cart add intent",
        )
        return "resolve_cart_product"
    if intent in _INTENTS_SKIP_HYBRID_CONTEXT:
        logger.debug("route_after_analyze_intent: skipping hybrid context for %s", intent)
        trace_route_decision(
            from_node="analyze_intent",
            target="call_mcp_tools",
            intent=intent,
            reason="intent skips hybrid context retrieval",
        )
        return "call_mcp_tools"
    if intent in ("discovery", "general"):
        user_message = _extract_latest_user_message(state.get("messages") or [])
        if contains_product_id(user_message):
            intent_metadata = state.get("intent_metadata") or {}
            has_city = bool(intent_metadata.get("target_city")) or bool(
                intent_metadata.get("requires_delivery_validation"),
            )
            if has_city:
                logger.debug(
                    "route_after_analyze_intent: product ID + city → resolve_delivery_context",
                )
                trace_route_decision(
                    from_node="analyze_intent",
                    target="resolve_delivery_context",
                    intent=intent,
                    reason="product ID with delivery city",
                )
                return "resolve_delivery_context"
            logger.debug(
                "route_after_analyze_intent: product ID fast-path for intent=%s",
                intent,
            )
            trace_route_decision(
                from_node="analyze_intent",
                target="call_mcp_tools",
                intent=intent,
                reason="product ID in message",
            )
            return "call_mcp_tools"
    trace_route_decision(
        from_node="analyze_intent",
        target="retrieve_hybrid_context",
        intent=intent,
        reason="discovery/general path",
    )
    return "retrieve_hybrid_context"


def route_after_master_flow(state: AgentState) -> RouteAfterAnalyzeIntent:
    """Route after master_flow: clarify short-circuit or normal post-intent routing."""
    master_clarifying = state.get("master_clarifying_question")
    if isinstance(master_clarifying, str) and master_clarifying.strip():
        trace_route_decision(
            from_node="master_flow",
            target="generate_response",
            intent=state.get("intent"),
            reason="master flow clarifying question",
        )
        return "generate_response"
    return route_after_analyze_intent(state)
