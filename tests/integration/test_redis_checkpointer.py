"""Integration tests for LangGraph Redis checkpointer persistence."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import fakeredis.aioredis
import pytest
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry

from graphs.shopping_graph import (
    append_message_state,
    build_shopping_graph,
    initial_shopping_state,
)
from lib.redis.checkpointer import get_checkpointer
from lib.redis.client import RedisClient

_THREAD_ID = "thread-checkpoint-test-001"
_SESSION_ID = "sess-checkpoint-test-001"


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    """Skip RediSearch index creation; fakeredis lacks FT._LIST."""
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


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
    graph = build_shopping_graph(checkpointer=checkpointer)
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

    second = await graph.ainvoke(append_message_state("something with chocolate"), config)
    assert second["tool_call_count"] == 2
    assert len(second["messages"]) == 2

    snapshot = await graph.aget_state(config)
    assert snapshot.values["tool_call_count"] == 2
    assert len(snapshot.values["messages"]) == 2


@pytest.mark.asyncio
async def test_different_thread_ids_have_isolated_state(
    checkpointer: AsyncRedisSaver,
) -> None:
    """Separate thread_id values do not share checkpointed state."""
    graph = build_shopping_graph(checkpointer=checkpointer)

    await graph.ainvoke(
        initial_shopping_state(message="first thread", session_id=_SESSION_ID),
        {"configurable": {"thread_id": "thread-a"}},
    )
    result_b = await graph.ainvoke(
        initial_shopping_state(message="second thread", session_id=_SESSION_ID),
        {"configurable": {"thread_id": "thread-b"}},
    )
    assert result_b["tool_call_count"] == 1
