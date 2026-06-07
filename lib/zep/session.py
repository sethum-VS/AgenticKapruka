"""Zep memory session create and resume backed by Redis."""

from __future__ import annotations

from typing import Final, cast

from lib.redis.client import RedisClient
from lib.zep.client import ZepClient

SESSION_TTL_SECONDS: Final = 7 * 24 * 60 * 60  # 7 days


def session_mapping_key(session_id: str) -> str:
    """Redis key mapping browser session_id to Zep thread_id."""
    return f"zep:session:{session_id}"


async def get_or_create_session(
    redis_client: RedisClient,
    zep_client: ZepClient,
    session_id: str,
) -> str:
    """Return Zep thread id for session_id, creating a Zep memory session on first visit."""
    key = session_mapping_key(session_id)
    existing = cast(str | None, await redis_client.client.get(key))
    if existing is not None:
        await redis_client.client.expire(key, SESSION_TTL_SECONDS)
        return existing

    zep_thread_id = session_id
    claimed = await redis_client.client.set(
        key,
        zep_thread_id,
        ex=SESSION_TTL_SECONDS,
        nx=True,
    )
    if claimed:
        await zep_client.create_session(zep_thread_id)
        return zep_thread_id

    existing = cast(str | None, await redis_client.client.get(key))
    if existing is not None:
        await redis_client.client.expire(key, SESSION_TTL_SECONDS)
        return existing

    await zep_client.create_session(zep_thread_id)
    await redis_client.client.set(key, zep_thread_id, ex=SESSION_TTL_SECONDS)
    return zep_thread_id
