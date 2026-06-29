"""Integration test: 2-turn Galle roses dialogue (clarify+search)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

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
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.types import (
    CategoryRef,
    CheckDeliveryOutput,
    Money,
    ProductResult,
    SearchProductsOutput,
)
from lib.redis.checkpointer import get_checkpointer
from lib.redis.client import RedisClient

_THREAD_ID = "thread-galle-roses-001"
_SESSION_ID = "sess-galle-roses-001"
_CLIENT_IP = "203.0.113.42"

_ROSE_PRODUCT = ProductResult(
    id="flower00ka001001",
    name="Romantic Red Rose Bouquet",
    summary="Fresh red roses arranged for delivery.",
    price=Money(amount=4500.0, currency="LKR"),
    compare_at_price=None,
    in_stock=True,
    stock_level="high",
    image_url="https://example.com/roses.jpg",
    category=CategoryRef(id="cat_flowers", name="Flowers", slug="flowers"),
    rating=None,
    ships_internationally=False,
    url="https://www.kapruka.com/roses",
)

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[_ROSE_PRODUCT],
    next_cursor=None,
    applied_filters={"q": "roses", "limit": 10, "in_stock_only": False},
)

_CHECK_DELIVERY_OUTPUT = CheckDeliveryOutput(
    city="Galle",
    now="2026-06-26T12:00:00+05:30",
    checked_date="2026-06-27",
    available=True,
    rate=1090.0,
    currency="LKR",
    reason=None,
    next_available_date=None,
    perishable_warning=None,
)


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
async def checkpointer(redis_client: RedisClient) -> AsyncRedisSaver:
    with patch.object(AsyncRedisSaver, "asetup", _fakeredis_asetup):
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
            response.parsed = AssistantReply(
                message="Here are romantic rose bouquets for Galle delivery.",
            )
            response.text = json.dumps(
                {"message": "Here are romantic rose bouquets for Galle delivery."},
            )
            return response
        response.parsed = IntentClassification(intent="discovery")
        response.text = json.dumps({"intent": "discovery"})
        return response

    mock_client.models.generate_content.side_effect = generate_content
    return mock_client


def _planner_steps_turn1() -> list[AgentPlannerStep]:
    return [
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "roses for Galle"},
            rationale="best-effort rose search while clarifying occasion",
        ),
        AgentPlannerStep(action="finish", rationale="search done"),
    ]


def _planner_steps_turn2() -> list[AgentPlannerStep]:
    return [
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "romantic red roses girlfriend birthday"},
            rationale="merged session rose search",
        ),
        AgentPlannerStep(action="finish", rationale="search done"),
    ]


@pytest.mark.asyncio
async def test_galle_roses_two_turn_clarify_and_carousel(
    checkpointer: AsyncRedisSaver,
) -> None:
    """Turn 1 clarifies occasion; turn 2 returns non-empty rose carousel."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    mock_service.list_delivery_cities.return_value = ["Galle"]
    mock_service.check_delivery.return_value = _CHECK_DELIVERY_OUTPUT

    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=_discovery_mock_genai(),
    )
    graph = build_shopping_graph(checkpointer=checkpointer, deps=deps)
    config: dict[str, Any] = {"configurable": {"thread_id": _THREAD_ID}}

    fixed = datetime(2026, 6, 26, 12, 0, tzinfo=ZoneInfo("Asia/Colombo"))
    with (
        patch("lib.utils.timezone.colombo_now", return_value=fixed),
        patch(
            "graphs.nodes.agent_loop._plan_next_step_sync",
            side_effect=_planner_steps_turn1(),
        ),
    ):
        turn1 = await graph.ainvoke(
            initial_shopping_state(
                message="Fresh roses to Galle tomorrow",
                session_id=_SESSION_ID,
                thread_id=_THREAD_ID,
            ),
            config,
        )

    assert turn1.get("agent_clarifying_question")
    search_calls = [
        inv for inv in (turn1.get("tool_trace") or []) if inv["name"] == SEARCH_PRODUCTS_TOOL
    ]
    assert search_calls, "turn 1 should still search despite occasion clarify"
    turn1_q = str(search_calls[0]["args"].get("q") or "")
    assert "Galle" not in turn1_q
    assert "rose" in turn1_q.lower()

    with (
        patch("lib.utils.timezone.colombo_now", return_value=fixed),
        patch(
            "graphs.nodes.agent_loop._plan_next_step_sync",
            side_effect=_planner_steps_turn2(),
        ),
    ):
        turn2 = await graph.ainvoke(
            append_message_state("Girlfriend's birthday, romantic red roses"),
            config,
        )

    products = turn2.get("last_search_products") or turn2.get("last_visible_products") or []
    assert products, "turn 2 carousel must be non-empty"
    names = " ".join(str(p.get("name") or "") for p in products if isinstance(p, dict)).lower()
    assert "rose" in names
    delivery_checks = [
        inv for inv in (turn2.get("tool_trace") or []) if inv["name"] == CHECK_DELIVERY_TOOL
    ]
    assert delivery_checks or mock_service.check_delivery.await_count >= 1
