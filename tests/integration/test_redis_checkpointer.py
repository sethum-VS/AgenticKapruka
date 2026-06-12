"""Integration tests for LangGraph Redis checkpointer persistence."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry
from tests.helpers.mock_genai import build_mock_genai_client

from graphs.shopping_graph import (
    ShoppingGraphDeps,
    append_message_state,
    build_shopping_graph,
    initial_shopping_state,
)
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import SearchProductsOutput, TrackOrderOutput
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
    assert second["tool_call_count"] == 2
    assert len(second["messages"]) == 2

    snapshot = await graph.aget_state(config)
    assert snapshot.values["tool_call_count"] == 2
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
    assert mock_client.models.generate_content.call_count == 5


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
    assert result_b["tool_call_count"] == 1
