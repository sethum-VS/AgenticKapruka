"""Unit tests for LangGraph Redis checkpointer setup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.exceptions

from lib.redis.checkpointer import get_checkpointer, redis_supports_redisearch
from lib.redis.client import RedisClient


@pytest.mark.asyncio
async def test_redis_supports_redisearch_false_on_unknown_command() -> None:
    client = MagicMock(spec=RedisClient)
    client.client = AsyncMock()
    client.client.execute_command.side_effect = redis.exceptions.ResponseError(
        "unknown command 'FT._LIST'"
    )

    assert await redis_supports_redisearch(client) is False


@pytest.mark.asyncio
async def test_get_checkpointer_returns_none_when_asetup_lacks_redisearch() -> None:
    client = MagicMock(spec=RedisClient)
    saver = MagicMock()
    saver.asetup = AsyncMock(
        side_effect=redis.exceptions.ResponseError("unknown command 'FT._LIST'")
    )

    with patch("lib.redis.checkpointer.create_checkpointer", return_value=saver):
        assert await get_checkpointer(client) is None


@pytest.mark.asyncio
async def test_get_checkpointer_reraises_unexpected_redis_errors() -> None:
    client = MagicMock(spec=RedisClient)
    saver = MagicMock()
    saver.asetup = AsyncMock(side_effect=redis.exceptions.ResponseError("OOM"))

    with (
        patch("lib.redis.checkpointer.create_checkpointer", return_value=saver),
        pytest.raises(redis.exceptions.ResponseError, match="OOM"),
    ):
        await get_checkpointer(client)


@pytest.mark.asyncio
async def test_get_checkpointer_initializes_when_asetup_succeeds() -> None:
    client = MagicMock(spec=RedisClient)
    saver = MagicMock()
    saver.asetup = AsyncMock()

    with patch("lib.redis.checkpointer.create_checkpointer", return_value=saver):
        result = await get_checkpointer(client)

    assert result is saver
    saver.asetup.assert_awaited_once()
