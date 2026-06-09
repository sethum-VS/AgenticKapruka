"""LangGraph Redis checkpointer for conversation continuity across requests."""

from __future__ import annotations

import logging

import redis.exceptions
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from lib.redis.client import RedisClient

logger = logging.getLogger(__name__)


def create_checkpointer(redis_client: RedisClient) -> AsyncRedisSaver:
    """Return an AsyncRedisSaver bound to the shared Redis connection pool."""
    return AsyncRedisSaver(redis_client=redis_client.client)


async def redis_supports_redisearch(redis_client: RedisClient) -> bool:
    """Return True when the Redis instance exposes RediSearch (Redis Stack)."""
    try:
        await redis_client.client.execute_command("FT._LIST")  # type: ignore[no-untyped-call]
    except redis.exceptions.ResponseError:
        return False
    else:
        return True


async def get_checkpointer(redis_client: RedisClient) -> AsyncRedisSaver | None:
    """Create and initialize the LangGraph Redis checkpointer when RediSearch is available.

    Standard Memorystore for Redis does not ship RediSearch modules. ``asetup()`` fails
    with ``unknown command 'FT._LIST'`` in that case; we return ``None`` so graphs compile
    without checkpoint persistence rather than failing every chat request.

    E2E and integration tests patch ``AsyncRedisSaver.asetup`` on fakeredis; the try/except
    path preserves that hook while still degrading gracefully in production.
    """
    checkpointer = create_checkpointer(redis_client)
    try:
        await checkpointer.asetup()
    except redis.exceptions.ResponseError as exc:
        if "FT._LIST" not in str(exc) and "unknown command" not in str(exc).lower():
            raise
        logger.warning(
            "RediSearch unavailable on REDIS_URL — LangGraph checkpointer disabled; "
            "deploy Redis Stack for multi-turn checkpoint persistence"
        )
        return None
    logger.debug("LangGraph Redis checkpointer initialized")
    return checkpointer
