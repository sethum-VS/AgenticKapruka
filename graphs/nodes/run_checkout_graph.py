"""Invoke the checkout sub-graph from the main shopping LangGraph."""

from __future__ import annotations

import logging
from typing import Any, cast

from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.checkout_graph import CheckoutGraphDeps, get_checkout_graph
from graphs.checkout_state import (
    CHECKOUT_STEP_ORDER,
    CheckoutState,
    initial_checkout_state,
    next_checkout_step,
)
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState, CheckoutStep
from lib.checkout.chat_parser import (
    apply_chat_message_to_checkout,
    prepare_checkout_invoke_state,
    should_auto_advance_step,
    should_chain_finalize,
)
from lib.checkout.prefill import seed_checkout_from_agent_state
from lib.checkout.review import review_context_from_checkout_state
from lib.chat.intent_heuristics import is_proceed_checkout_message
from lib.kapruka.service import KaprukaService
from lib.redis.cart import get_cart
from lib.redis.checkout import get_checkout_session, save_checkout_session
from lib.redis.client import RedisClient

logger = logging.getLogger(__name__)


async def load_cart_items_for_checkout(
    redis_client: RedisClient | None,
    session_id: str,
) -> list[dict[str, Any]]:
    """Hydrate checkout cart_items from the session Redis cart."""
    if not redis_client or not session_id:
        return []
    rows = await get_cart(redis_client, session_id)
    return [row.model_dump() for row in rows]


def _merge_persisted_into_checkout(
    checkout_input: CheckoutState,
    persisted: dict[str, Any],
) -> CheckoutState:
    """Overlay Redis checkout session onto freshly hydrated cart state."""
    merged = dict(checkout_input)
    for key, value in persisted.items():
        if value is not None:
            merged[key] = value
    return cast(CheckoutState, merged)


def _maybe_render_review_html(state: CheckoutState) -> str | None:
    """Render review partial when all pre-review steps are valid."""
    if state.get("current_step") != "review":
        return None
    context = review_context_from_checkout_state(state)
    if context is None:
        return None
    from app.templating import render_checkout_review

    return render_checkout_review(review=context)


def _checkout_tool_payload(
    result: CheckoutState,
    *,
    cart_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build run_checkout_graph tool_results payload from checkout state."""
    current_step = result.get("current_step")
    resolved_cart_items: list[dict[str, Any]] = list(result.get("cart_items") or cart_items)
    review_html = (
        result.get("response_html")
        if current_step == "review"
        else _maybe_render_review_html(result)
    )
    return {
        "current_step": current_step,
        "cart_items": resolved_cart_items,
        "step_valid": result.get("step_valid") or {},
        "validation_errors": result.get("validation_errors"),
        "delivery_city": result.get("delivery_city"),
        "delivery_address": result.get("delivery_address"),
        "delivery_location_type": result.get("delivery_location_type"),
        "delivery_date": result.get("delivery_date"),
        "delivery_instructions": result.get("delivery_instructions"),
        "recipient_name": result.get("recipient_name"),
        "recipient_phone": result.get("recipient_phone"),
        "sender_name": result.get("sender_name"),
        "sender_anonymous": result.get("sender_anonymous"),
        "gift_message": result.get("gift_message"),
        "review_html": review_html if current_step == "review" else None,
        "payment_cta_html": result.get("response_html") if current_step == "finalize" else None,
        "checkout_url": result.get("checkout_url"),
        "order_ref": result.get("order_ref"),
        "expires_at": result.get("expires_at"),
        "order_summary": result.get("order_summary"),
    }


async def _invoke_checkout_graph(
    graph: Any,
    state: CheckoutState,
) -> dict[str, Any]:
    """Run checkout sub-graph, chaining finalize when review advances."""
    merged = dict(state)
    result = cast(dict[str, Any], await graph.ainvoke(merged))
    prev_step = cast(CheckoutStep, merged.get("current_step") or "cart")
    merged = {**merged, **result}

    new_step = cast(CheckoutStep, merged.get("current_step") or prev_step)
    if should_chain_finalize(prev_step, new_step, merged):
        merged["action"] = "advance"
        merged["target_step"] = "finalize"
        result = cast(dict[str, Any], await graph.ainvoke(merged))
        merged = {**merged, **result}

    auto_guard = 0
    while should_auto_advance_step(cast(CheckoutState, merged)) and auto_guard < len(
        CHECKOUT_STEP_ORDER,
    ):
        auto_guard += 1
        current = cast(CheckoutStep, merged.get("current_step") or "cart")
        nxt = next_checkout_step(current)
        if nxt is None:
            break
        merged["action"] = "advance"
        merged["target_step"] = nxt
        prev = current
        result = cast(dict[str, Any], await graph.ainvoke(merged))
        merged = {**merged, **result}
        new_step = cast(CheckoutStep, merged.get("current_step") or prev)
        if should_chain_finalize(prev, new_step, merged):
            merged["action"] = "advance"
            merged["target_step"] = "finalize"
            result = cast(dict[str, Any], await graph.ainvoke(merged))
            merged = {**merged, **result}
            break

    return merged


async def run_checkout_graph(
    state: AgentState,
    *,
    redis_client: RedisClient | None = None,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
) -> dict[str, Any]:
    """LangGraph node: run checkout sub-graph with Redis cart and session hydration."""
    session_id = state.get("session_id") or ""
    currency = state.get("currency")

    user_message = _extract_latest_user_message(state.get("messages") or [])
    if state.get("checkout_paused") and not is_proceed_checkout_message(user_message):
        checkout_step = state.get("checkout_state") or "cart"
        cart_items = await load_cart_items_for_checkout(redis_client, session_id)
        checkout_input = initial_checkout_state(
            session_id=session_id,
            currency=currency if currency is not None else None,
            cart_items=cart_items,
        )
        if redis_client is not None and session_id:
            persisted = await get_checkout_session(redis_client, session_id)
            checkout_input = _merge_persisted_into_checkout(checkout_input, persisted)
        checkout_input = seed_checkout_from_agent_state(checkout_input, state)
        active_step = checkout_input.get("current_step") or checkout_step
        payload = _checkout_tool_payload(checkout_input, cart_items=cart_items)
        logger.info(
            "run_checkout_graph: checkout paused — holding at step %s",
            active_step,
        )
        return {
            "checkout_state": active_step,
            "checkout_paused": True,
            "tool_results": {CHECKOUT_TOOL_KEY: payload},
        }

    cart_items = await load_cart_items_for_checkout(redis_client, session_id)
    checkout_input = initial_checkout_state(
        session_id=session_id,
        currency=currency if currency is not None else None,
        cart_items=cart_items,
    )

    if redis_client is not None and session_id:
        persisted = await get_checkout_session(redis_client, session_id)
        checkout_input = _merge_persisted_into_checkout(checkout_input, persisted)

    checkout_input = seed_checkout_from_agent_state(checkout_input, state)

    user_message = _extract_latest_user_message(state.get("messages") or [])
    active_step = checkout_input.get("current_step") or "cart"
    resume_from_pause = bool(state.get("checkout_paused")) and is_proceed_checkout_message(
        user_message,
    )
    if (
        user_message.strip()
        and is_proceed_checkout_message(user_message)
        and active_step != "cart"
        and not resume_from_pause
    ):
        logger.info(
            "run_checkout_graph: skipping duplicate proceed-to-checkout at step %s",
            active_step,
        )
        payload = _checkout_tool_payload(checkout_input, cart_items=cart_items)
        return {
            "checkout_state": active_step,
            "tool_results": {CHECKOUT_TOOL_KEY: payload},
        }

    if user_message.strip():
        checkout_input = apply_chat_message_to_checkout(checkout_input, user_message)

    checkout_input = prepare_checkout_invoke_state(checkout_input)

    deps = CheckoutGraphDeps(
        redis_client=redis_client,
        kapruka_service=kapruka_service,
        client_ip=client_ip or "127.0.0.1",
    )
    graph = get_checkout_graph(deps=deps)
    result = await _invoke_checkout_graph(graph, checkout_input)

    current_step = result.get("current_step")
    resolved_cart_items: list[dict[str, Any]] = list(result.get("cart_items") or cart_items)
    payload = _checkout_tool_payload(
        cast(CheckoutState, result),
        cart_items=resolved_cart_items,
    )

    if redis_client is not None and session_id:
        persist_state = cast(CheckoutState, {**result, "cart_items": resolved_cart_items})
        await save_checkout_session(redis_client, session_id, persist_state)
        if current_step == "finalize" and result.get("checkout_url"):
            from lib.redis.checkout import clear_checkout_session

            await clear_checkout_session(redis_client, session_id)

    logger.info(
        "run_checkout_graph: session=%s step=%s cart_lines=%d",
        session_id,
        current_step,
        len(resolved_cart_items),
    )

    updates: dict[str, Any] = {
        "checkout_state": current_step,
        "tool_results": {CHECKOUT_TOOL_KEY: payload},
        "checkout_paused": False,
    }
    if current_step in ("review", "finalize"):
        updates["model_tier"] = "pro"

    return updates
