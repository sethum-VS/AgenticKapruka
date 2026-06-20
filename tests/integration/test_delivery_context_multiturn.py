"""Integration tests for session-scoped delivery city across multi-turn chat."""

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

_THREAD_ID = "thread-delivery-multiturn-001"
_SESSION_ID = "sess-delivery-multiturn-001"
_CLIENT_IP = "203.0.113.42"

_CAKE_PRODUCT = ProductResult(
    id="cake00ka002034",
    name="Chocolate Birthday Cake",
    summary="Rich chocolate layers.",
    price=Money(amount=4500.0, currency="LKR"),
    compare_at_price=None,
    in_stock=True,
    stock_level="high",
    image_url="https://example.com/cake.jpg",
    category=CategoryRef(id="cat_cakes", name="Birthday", slug="birthday"),
    rating=None,
    ships_internationally=False,
    url="https://www.kapruka.com/cake",
)

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[_CAKE_PRODUCT],
    next_cursor=None,
    applied_filters={"q": "birthday cake", "limit": 10, "in_stock_only": False},
)

_CHECK_DELIVERY_OUTPUT = CheckDeliveryOutput(
    city="Kandy",
    now="2026-06-12T12:00:00+05:30",
    checked_date="2026-06-13",
    available=True,
    rate=500.0,
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
            response.parsed = AssistantReply(message="Here are some Kapruka options.")
            response.text = json.dumps({"message": "Here are some Kapruka options."})
            return response
        response.parsed = IntentClassification(intent="discovery")
        response.text = json.dumps({"intent": "discovery"})
        return response

    mock_client.models.generate_content.side_effect = generate_content
    return mock_client


def _planner_steps_for_multiturn() -> list[AgentPlannerStep]:
    return [
        AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": "birthday cake"},
            rationale="search cakes",
        ),
        AgentPlannerStep(action="finish", rationale="cakes found"),
        AgentPlannerStep(
            action="call_tool",
            tool_name=CHECK_DELIVERY_TOOL,
            tool_args={"city": "Kandy"},
            rationale="check delivery",
        ),
        AgentPlannerStep(
            action="call_tool",
            tool_name=CHECK_DELIVERY_TOOL,
            tool_args={"city": "Kandy"},
            rationale="check delivery with date",
        ),
        AgentPlannerStep(action="finish", rationale="delivery checked"),
    ]


@pytest.mark.asyncio
async def test_kandy_cake_deliver_tomorrow_multiturn_check_delivery(
    checkpointer: AsyncRedisSaver,
) -> None:
    """Eval B-01: turn 3 uses session city Kandy for kapruka_check_delivery after date reply."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    mock_service.list_delivery_cities.return_value = ["Kandy"]
    mock_service.check_delivery.return_value = _CHECK_DELIVERY_OUTPUT

    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=_discovery_mock_genai(),
    )
    graph = build_shopping_graph(checkpointer=checkpointer, deps=deps)
    config: dict[str, Any] = {"configurable": {"thread_id": _THREAD_ID}}
    planner_steps = _planner_steps_for_multiturn()

    fixed = datetime(2026, 6, 12, 12, 0, tzinfo=ZoneInfo("Asia/Colombo"))
    with (
        patch("lib.utils.timezone.colombo_now", return_value=fixed),
        patch(
            "graphs.nodes.agent_loop._plan_next_step_sync",
            side_effect=planner_steps,
        ),
    ):
        turn1 = await graph.ainvoke(
            initial_shopping_state(
                message="birthday cake for mom in Kandy",
                session_id=_SESSION_ID,
                thread_id=_THREAD_ID,
            ),
            config,
        )
        assert turn1.get("session_delivery_city_canonical") == "Kandy"

        turn2 = await graph.ainvoke(append_message_state("can you deliver?"), config)
        assert turn2.get("session_awaiting_delivery_date") is True
        mock_service.check_delivery.assert_not_awaited()

        turn3 = await graph.ainvoke(append_message_state("tomorrow"), config)

    tool_trace = turn3.get("tool_trace") or []
    check_calls = [inv for inv in tool_trace if inv["name"] == CHECK_DELIVERY_TOOL]
    assert check_calls, "turn 3 must invoke kapruka_check_delivery"
    assert check_calls[-1]["args"]["city"] == "Kandy"
    assert check_calls[-1]["args"]["delivery_date"] == "2026-06-13"
    assert turn3.get("session_awaiting_delivery_date") is False

    mock_service.check_delivery.assert_awaited_once()
    await_args = mock_service.check_delivery.await_args
    assert await_args is not None
    assert await_args.args[0] == _CLIENT_IP
    assert await_args.kwargs["city"] == "Kandy"
    assert await_args.kwargs["delivery_date"] == "2026-06-13"
