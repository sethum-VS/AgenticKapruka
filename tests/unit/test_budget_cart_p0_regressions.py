"""Regression: mid-chat budget refine must filter prior carousel even without MCP args."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from graphs.nodes.agent_loop import _try_budget_refinement_fast_path
from lib.kapruka.service import KaprukaService


def _choc(product_id: str, name: str, amount: float) -> dict[str, Any]:
    return {
        "id": product_id,
        "name": name,
        "price": {"amount": amount, "currency": "LKR"},
        "in_stock": True,
        "category": {"name": "Chocolate", "slug": "chocolate"},
    }


@pytest.mark.asyncio
async def test_budget_refine_in_memory_when_search_args_unavailable() -> None:
    """Prior under-budget carousel wins even if build_budget_refinement_search_args is None."""
    prior = [
        _choc("over", "Luxury Chocolate Tower", 9580.0),
        _choc("under_a", "Vibe Check Chocolate Gift Box", 4100.0),
        _choc("under_b", "Fruits Chocolates Harmony", 4900.0),
    ]
    state: dict[str, Any] = {
        "messages": [{"content": "Under 6000"}],  # placeholder; patched below
        "session_product_focus": None,
        "session_search_query": None,
        "last_visible_products": prior,
        "last_search_products": prior,
        "session_budget_max": None,
        "intent_metadata": {"budgeted_gift_discovery": True},
        "tool_call_count": 0,
        "currency": "LKR",
    }
    from langchain_core.messages import HumanMessage

    state["messages"] = [HumanMessage(content="Under 6000")]

    service = AsyncMock(spec=KaprukaService)
    result = await _try_budget_refinement_fast_path(
        state,  # type: ignore[arg-type]
        tool_trace=[],
        kapruka_service=service,
        rate_limit_key="127.0.0.1",
        currency="LKR",
    )
    assert result is not None
    products = result.get("last_search_products") or []
    ids = {p.get("id") for p in products}
    assert "under_a" in ids
    assert "under_b" in ids
    assert "over" not in ids
    service.search_products.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_that_uses_session_resolved_when_carousel_missing() -> None:
    """Deictic cart add falls back to session_resolved_product if lists were cleared."""
    from langchain_core.messages import HumanMessage

    from graphs.nodes.resolve_cart_product import resolve_cart_product

    product = _choc("fallback1", "Kunafa Chocolate", 1070.0)
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="Add that to my cart.")],
        "last_visible_products": None,
        "last_search_products": None,
        "session_resolved_product": product,
        "session_product_focus": "chocolate",
        "currency": "LKR",
    }
    out = await resolve_cart_product(state)  # type: ignore[arg-type]
    action = out.get("cart_action_result") or {}
    assert action.get("status") == "resolved"
    assert (action.get("product") or {}).get("id") == "fallback1"
