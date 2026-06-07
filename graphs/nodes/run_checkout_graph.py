"""Invoke the checkout sub-graph from the main shopping LangGraph."""

from __future__ import annotations

import logging
from typing import Any

from graphs.checkout_constants import CHECKOUT_TOOL_KEY
from graphs.checkout_graph import CheckoutGraphDeps, get_checkout_graph
from graphs.checkout_state import initial_checkout_state
from graphs.state import AgentState
from lib.kapruka.service import KaprukaService
from lib.redis.cart import get_cart
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


async def run_checkout_graph(
    state: AgentState,
    *,
    redis_client: RedisClient | None = None,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
) -> dict[str, Any]:
    """LangGraph node: run checkout sub-graph with Redis cart hydration."""
    session_id = state.get("session_id") or ""
    currency = state.get("currency")

    cart_items = await load_cart_items_for_checkout(redis_client, session_id)
    checkout_input = initial_checkout_state(
        session_id=session_id,
        currency=currency if currency is not None else None,
        cart_items=cart_items,
    )

    deps = CheckoutGraphDeps(
        redis_client=redis_client,
        kapruka_service=kapruka_service,
        client_ip=client_ip or "127.0.0.1",
    )
    graph = get_checkout_graph(deps=deps)
    result = await graph.ainvoke(checkout_input)

    current_step = result.get("current_step")
    resolved_cart_items: list[dict[str, Any]] = list(result.get("cart_items") or [])
    payload = {
        "current_step": current_step,
        "cart_items": resolved_cart_items,
        "step_valid": result.get("step_valid") or {},
        "validation_errors": result.get("validation_errors"),
    }

    logger.info(
        "run_checkout_graph: session=%s step=%s cart_lines=%d",
        session_id,
        current_step,
        len(resolved_cart_items),
    )

    return {
        "checkout_state": current_step,
        "tool_results": {CHECKOUT_TOOL_KEY: payload},
    }
