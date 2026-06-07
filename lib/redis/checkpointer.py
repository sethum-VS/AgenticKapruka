"""LangGraph Redis checkpointer for conversation continuity across requests."""

from __future__ import annotations

import logging

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from lib.redis.client import RedisClient

logger = logging.getLogger(__name__)


def create_checkpointer(redis_client: RedisClient) -> AsyncRedisSaver:
    """Return an AsyncRedisSaver bound to the shared Redis connection pool."""
    return AsyncRedisSaver(redis_client=redis_client.client)


async def get_checkpointer(redis_client: RedisClient) -> AsyncRedisSaver:
    """Create and initialize the LangGraph Redis checkpointer.

    Calls ``asetup()`` to create RediSearch indices required by Memorystore /
    Redis Stack. Must be invoked once per process before compiling graphs.
    """
    checkpointer = create_checkpointer(redis_client)
    await checkpointer.asetup()
    logger.debug("LangGraph Redis checkpointer initialized")
    return checkpointer
