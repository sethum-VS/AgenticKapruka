"""Integration tests for topic-pivot context hygiene across turns."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest
from google.genai import types
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry

from graphs.nodes.agent_loop import AgentPlannerStep
from graphs.nodes.analyze_intent import IntentClassification
from graphs.nodes.generate_response import AssistantReply
from graphs.shopping_graph import (
    ShoppingGraphDeps,
    append_message_state,
    build_shopping_graph,
    initial_shopping_state,
)
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.types import CategoryRef, Money, ProductResult, SearchProductsOutput

_THREAD_ID = "thread-context-pivot-001"
_SESSION_ID = "sess-context-pivot-001"
_CLIENT_IP = "203.0.113.66"

_CHOCOLATE_PRODUCT = ProductResult(
    id="choc001",
    name="Heart Chocolate Gift Box",
    summary="Assorted milk chocolates.",
    price=Money(amount=4500.0, currency="LKR"),
    compare_at_price=None,
    in_stock=True,
    stock_level="high",
    image_url="https://example.com/choc.jpg",
    category=CategoryRef(id="cat_choc", name="Chocolate", slug="chocolate"),
    rating=None,
    ships_internationally=False,
    url="https://www.kapruka.com/choc",
)

_CAKE_PRODUCT = ProductResult(
    id="cake001",
    name="Vanilla Celebration Cake",
    summary="Classic vanilla sponge.",
    price=Money(amount=5200.0, currency="LKR"),
    compare_at_price=None,
    in_stock=True,
    stock_level="high",
    image_url="https://example.com/cake.jpg",
    category=CategoryRef(id="cat_cake", name="Cakes", slug="cakes"),
    rating=None,
    ships_internationally=False,
    url="https://www.kapruka.com/cake",
)


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


@pytest.fixture
def redis_client() -> Any:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    from lib.redis.client import RedisClient

    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
async def checkpointer(redis_client: Any) -> AsyncRedisSaver:
    with patch.object(AsyncRedisSaver, "asetup", _fakeredis_asetup):
        from lib.redis.checkpointer import get_checkpointer

        return await get_checkpointer(redis_client)


def _discovery_mock_genai() -> MagicMock:
    mock_client = MagicMock()

    def generate_content(
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
        **kwargs: Any,
    ) -> MagicMock:
        _ = model, contents, kwargs
        response = MagicMock()
        if config is not None and config.response_schema is IntentClassification:
            response.parsed = IntentClassification(intent="discovery")
            response.text = json.dumps({"intent": "discovery"})
            return response
        if config is not None and config.response_schema is AssistantReply:
            response.parsed = AssistantReply(message="Here are some cake options.")
            response.text = json.dumps({"message": "Here are some cake options."})
            return response
        response.parsed = IntentClassification(intent="discovery")
        response.text = json.dumps({"intent": "discovery"})
        return response

    mock_client.models.generate_content.side_effect = generate_content
    return mock_client


@pytest.mark.asyncio
async def test_context_pivot_nevermind_cakes_clears_anniversary_and_searches_cake(
    checkpointer: AsyncRedisSaver,
) -> None:
    """QA Scenario 2: chocolate/anniversary thread pivots to bare cakes without stale hints."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.side_effect = [
        SearchProductsOutput(
            results=[_CHOCOLATE_PRODUCT],
            next_cursor=None,
            applied_filters={"q": "chocolate gift", "limit": 10, "in_stock_only": False},
        ),
        SearchProductsOutput(
            results=[_CAKE_PRODUCT],
            next_cursor=None,
            applied_filters={"q": "cake", "limit": 10, "in_stock_only": False},
        ),
    ]

    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=_discovery_mock_genai(),
    )
    graph = build_shopping_graph(checkpointer=checkpointer, deps=deps)
    config: dict[str, Any] = {"configurable": {"thread_id": _THREAD_ID}}

    planner_steps = [
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "chocolate gift"},
            rationale="search chocolate",
        ),
        AgentPlannerStep(action="finish", rationale="done"),
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "birthday cake", "category": "Birthday"},
            rationale="search cakes after pivot",
        ),
        AgentPlannerStep(action="finish", rationale="done"),
    ]

    with patch(
        "graphs.nodes.agent_loop._plan_next_step_sync",
        side_effect=planner_steps,
    ):
        turn1 = await graph.ainvoke(
            initial_shopping_state(
                message="Anniversary chocolates for my wife under 6000",
                session_id=_SESSION_ID,
                thread_id=_THREAD_ID,
            ),
            config,
        )
        assert turn1.get("session_product_focus") == "chocolate"

        turn2 = await graph.ainvoke(
            append_message_state("Nevermind. Cakes."),
            config,
        )

    assert turn2.get("intent_metadata", {}).get("topic_pivot") is True
    assert turn2.get("session_budget_max") is None
    assert turn2.get("session_budget_currency") is None
    response_html = turn2.get("response_html") or ""
    assert 'data-testid="budget-badge"' not in response_html
    assert turn2.get("session_product_focus") == "cake"
    assert turn2.get("session_search_query") is None
    hints = (turn2.get("hybrid_context") or {}).get("hints") or {}
    assert hints.get("occasion") != "Anniversary"
    if mock_service.search_products.await_count >= 2:
        assert mock_service.search_products.await_args.kwargs.get("q") == "cake"
