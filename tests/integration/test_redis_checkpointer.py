"""Integration tests for LangGraph Redis checkpointer persistence."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest
from google.genai import types
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry
from tests.helpers.mock_genai import build_mock_genai_client

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
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import (
    CategoryRef,
    Money,
    ProductResult,
    SearchProductsOutput,
    TrackOrderOutput,
)
from lib.redis.checkpointer import get_checkpointer
from lib.redis.client import RedisClient

_THREAD_ID = "thread-checkpoint-test-001"
_SESSION_ID = "sess-checkpoint-test-001"
_CLIENT_IP = "203.0.113.42"

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[],
    next_cursor=None,
    applied_filters={"q": "birthday cake", "limit": 10, "in_stock_only": False},
)

_TRACK_OUTPUT = TrackOrderOutput.model_validate(
    {
        "order_number": "VIMP34456CB2",
        "pnref": "12345678901",
        "status": "shipped",
        "status_display": "Out for Delivery",
        "order_date": "June 5, 2026",
        "delivery_date": "June 7, 2026",
        "shipped_date": "June 6, 2026",
        "amount": "15500.00",
        "payment_method": "Visa",
        "comments": None,
        "recipient": {
            "name": "Ada Lovelace",
            "phone": "0771234567",
            "address": "123 Galle Road",
            "city": "Colombo 03",
        },
        "greeting_message": None,
        "special_instructions": None,
        "progress": [{"step": "shipped", "timestamp": "June 6, 2026 08:00 AM"}],
        "live_tracking_available": False,
        "has_delivery_video": False,
        "has_delivery_photo": False,
        "items": [],
    },
)


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    """Skip RediSearch index creation; fakeredis lacks FT._LIST."""
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


def _checkpoint_graph_deps() -> ShoppingGraphDeps:
    """Mocks for full graph runs in checkpoint persistence tests."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT

    return ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=build_mock_genai_client(
            search_query="birthday cake",
            assistant_message="Here are some birthday cake options.",
        ),
    )


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
async def checkpointer(redis_client: RedisClient) -> AsyncRedisSaver:
    with patch.object(AsyncRedisSaver, "asetup", _fakeredis_asetup):
        return await get_checkpointer(redis_client)


@pytest.mark.asyncio
async def test_state_persists_across_two_invocations_same_thread_id(
    checkpointer: AsyncRedisSaver,
) -> None:
    """Checkpoint restores prior graph state when re-invoked with the same thread_id."""
    graph = build_shopping_graph(checkpointer=checkpointer, deps=_checkpoint_graph_deps())
    config: dict[str, Any] = {"configurable": {"thread_id": _THREAD_ID}}

    first = await graph.ainvoke(
        initial_shopping_state(
            message="birthday cake for mom",
            session_id=_SESSION_ID,
            thread_id=_THREAD_ID,
        ),
        config,
    )
    assert first["tool_call_count"] == 1
    assert len(first["messages"]) == 1
    assert SEARCH_PRODUCTS_TOOL in (first.get("tool_results") or {})

    second = await graph.ainvoke(append_message_state("something with chocolate"), config)
    assert len(second["messages"]) == 2
    assert second.get("agent_clarifying_question") is None

    snapshot = await graph.aget_state(config)
    assert len(snapshot.values["messages"]) == 2


@pytest.mark.asyncio
async def test_tracking_turn_clears_stale_tool_results_from_prior_discovery(
    checkpointer: AsyncRedisSaver,
) -> None:
    """Follow-up tracking must not retain discovery MCP payloads from checkpoint."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    mock_service.track_order.return_value = _TRACK_OUTPUT

    mock_client = build_mock_genai_client(
        intent=["discovery", "tracking"],
        search_query="birthday cake",
        assistant_message="Here are some birthday cake options.",
    )

    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=mock_client,
    )
    graph = build_shopping_graph(checkpointer=checkpointer, deps=deps)
    config: dict[str, Any] = {"configurable": {"thread_id": _THREAD_ID}}

    first = await graph.ainvoke(
        initial_shopping_state(
            message="birthday cake for mom",
            session_id=_SESSION_ID,
            thread_id=_THREAD_ID,
        ),
        config,
    )
    assert SEARCH_PRODUCTS_TOOL in (first.get("tool_results") or {})

    second = await graph.ainvoke(
        append_message_state("where is order VIMP34456CB2"),
        config,
    )
    assert second["intent"] == "tracking"
    second_results = second.get("tool_results") or {}
    assert TRACK_ORDER_TOOL in second_results
    assert SEARCH_PRODUCTS_TOOL not in second_results
    mock_service.track_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_tracking_turn_with_money_shaped_amount_renders_without_crash(
    checkpointer: AsyncRedisSaver,
) -> None:
    """MCP value/currency amount objects must not crash tracking card rendering."""
    money_track = TrackOrderOutput.model_validate(
        {
            **_TRACK_OUTPUT.model_dump(),
            "amount": {"value": "4970", "currency": "LKR"},
        }
    )
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.track_order.return_value = money_track

    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=build_mock_genai_client(
            intent=["tracking"],
            assistant_message="Here is your order status.",
        ),
    )
    graph = build_shopping_graph(checkpointer=checkpointer, deps=deps)
    config: dict[str, Any] = {"configurable": {"thread_id": "thread-money-amount"}}

    result = await graph.ainvoke(
        initial_shopping_state(
            message="Track order VIMP34456CB2",
            session_id=_SESSION_ID,
            thread_id="thread-money-amount",
        ),
        config,
    )

    assert result["intent"] == "tracking"
    assert 'data-testid="order-tracking-status"' in (result.get("response_html") or "")
    assert "LKR 4,970" in (result.get("response_html") or "")
    mock_service.track_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_different_thread_ids_have_isolated_state(
    checkpointer: AsyncRedisSaver,
) -> None:
    """Separate thread_id values do not share checkpointed state."""
    graph = build_shopping_graph(checkpointer=checkpointer, deps=_checkpoint_graph_deps())

    await graph.ainvoke(
        initial_shopping_state(message="first thread", session_id=_SESSION_ID),
        {"configurable": {"thread_id": "thread-a"}},
    )
    result_b = await graph.ainvoke(
        initial_shopping_state(message="second thread", session_id=_SESSION_ID),
        {"configurable": {"thread_id": "thread-b"}},
    )
    snap_a = await graph.aget_state({"configurable": {"thread_id": "thread-a"}})
    snap_b = await graph.aget_state({"configurable": {"thread_id": "thread-b"}})
    assert snap_a.values["messages"][0].content == "first thread"
    assert snap_b.values["messages"][0].content == "second thread"
    assert result_b["messages"][0].content == "second thread"


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


def _ask_then_cakes_mock_genai() -> MagicMock:
    """Planner: turn 1 ask_user on vague gifts; turn 2 search cakes + finish."""
    mock_client = MagicMock()
    planner_calls = 0

    def generate_content(
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
        **kwargs: Any,
    ) -> MagicMock:
        nonlocal planner_calls
        _ = model, kwargs
        response = MagicMock()
        if config is not None and config.response_schema is IntentClassification:
            response.parsed = IntentClassification(intent="discovery")
            response.text = json.dumps({"intent": "discovery"})
            return response

        if config is not None and config.response_schema is AgentPlannerStep:
            planner_calls += 1
            if planner_calls == 1:
                step = AgentPlannerStep(
                    action="ask_user",
                    rationale="Who is the gift for or what occasion?",
                    refined_intent="discovery",
                )
            elif planner_calls == 2:
                step = AgentPlannerStep(
                    action="call_tool",
                    tool_name=SEARCH_PRODUCTS_TOOL,
                    tool_args={"q": "cakes"},
                    refined_intent="discovery",
                    rationale="search cakes",
                )
            else:
                step = AgentPlannerStep(action="finish", rationale="done")
            response.parsed = step
            response.text = step.model_dump_json()
            return response

        if config is not None and config.response_schema is AssistantReply:
            response.parsed = AssistantReply(message="Here are some cakes from Kapruka.")
            response.text = json.dumps({"message": "Here are some cakes from Kapruka."})
            return response

        response.parsed = IntentClassification(intent="discovery")
        response.text = json.dumps({"intent": "discovery"})
        return response

    mock_client.models.generate_content.side_effect = generate_content
    return mock_client


@pytest.mark.asyncio
async def test_cakes_after_ask_user_renders_carousel_not_stale_clarifying(
    checkpointer: AsyncRedisSaver,
) -> None:
    """Regression: stale agent_clarifying_question must not mask cakes search results."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = SearchProductsOutput(
        results=[_CAKE_PRODUCT],
        next_cursor=None,
        applied_filters={"q": "cakes", "limit": 10, "in_stock_only": False},
    )

    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=_ask_then_cakes_mock_genai(),
    )
    graph = build_shopping_graph(checkpointer=checkpointer, deps=deps)
    config: dict[str, Any] = {"configurable": {"thread_id": "thread-cakes-regression"}}

    first = await graph.ainvoke(
        initial_shopping_state(
            message="show me gifts",
            session_id=_SESSION_ID,
            thread_id="thread-cakes-regression",
        ),
        config,
    )
    assert first.get("agent_loop_exit_reason") == "ask_user"
    assert first.get("agent_clarifying_question")

    second = await graph.ainvoke(append_message_state("cakes"), config)
    html = second.get("response_html") or ""
    assert 'data-testid="product-carousel"' in html
    assert 'data-product-id="cake00ka002034"' in html
    assert "previous search for 'gifts'" not in (second.get("assistant_message") or "").lower()
