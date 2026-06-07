"""Integration tests for LangGraph Redis checkpointer persistence."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry

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
from lib.kapruka.types import SearchProductsOutput
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


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    """Skip RediSearch index creation; fakeredis lacks FT._LIST."""
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


def _checkpoint_graph_deps() -> ShoppingGraphDeps:
    """Mocks for full graph runs in checkpoint persistence tests."""
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT

    mock_client = MagicMock()
    intent_response = MagicMock()
    intent_response.parsed = IntentClassification(intent="discovery")
    intent_response.text = '{"intent": "discovery"}'
    reply_response = MagicMock()
    reply_response.parsed = AssistantReply(message="Here are some birthday cake options.")
    reply_response.text = reply_response.parsed.model_dump_json()
    mock_client.models.generate_content.side_effect = [
        intent_response,
        reply_response,
        intent_response,
        reply_response,
    ]

    return ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip=_CLIENT_IP,
        genai_client=mock_client,
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

    mock_client = MagicMock()
    discovery_intent = MagicMock()
    discovery_intent.parsed = IntentClassification(intent="discovery")
    discovery_intent.text = '{"intent": "discovery"}'
    discovery_reply = MagicMock()
    discovery_reply.parsed = AssistantReply(message="Here are some birthday cake options.")
    discovery_reply.text = discovery_reply.parsed.model_dump_json()

    tracking_intent = MagicMock()
    tracking_intent.parsed = IntentClassification(intent="tracking")
    tracking_intent.text = '{"intent": "tracking"}'
    tracking_reply = MagicMock()
    tracking_reply.parsed = AssistantReply(
        message="Please share your Kapruka order number so I can look up delivery status.",
    )
    tracking_reply.text = tracking_reply.parsed.model_dump_json()

    mock_client.models.generate_content.side_effect = [
        discovery_intent,
        discovery_reply,
        tracking_intent,
        tracking_reply,
    ]

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
    assert second.get("tool_results") == {}

    tracking_response_call = mock_client.models.generate_content.call_args_list[3]
    tracking_prompt = tracking_response_call.kwargs.get("contents")
    if tracking_prompt is None and len(tracking_response_call.args) > 1:
        tracking_prompt = tracking_response_call.args[1]
    assert "Chocolate Birthday Cake" not in str(tracking_prompt)
    assert '"results"' not in str(tracking_prompt)


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
