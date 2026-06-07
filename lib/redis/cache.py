"""Read-only Kapruka MCP response cache backed by Redis."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, cast

from lib.redis.client import RedisClient
from lib.redis.rate_limit import CREATE_ORDER_TOOL

DEFAULT_CACHE_TTL: Final = 1800


def is_cacheable_tool(tool: str) -> bool:
    """Return False for write tools (e.g. create_order); reads are cacheable."""
    return tool != CREATE_ORDER_TOOL


def canonical_args_json(args: dict[str, Any]) -> str:
    """Serialize args with sorted keys for stable cache-key hashing."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(tool: str, args: dict[str, Any]) -> str:
    """Build Redis key as tool:sha256(canonical_json_args)."""
    digest = hashlib.sha256(canonical_args_json(args).encode()).hexdigest()
    return f"{tool}:{digest}"


async def get_cached(
    redis_client: RedisClient,
    tool: str,
    args: dict[str, Any],
) -> str | None:
    """Return cached MCP response string, or None on miss or non-cacheable tool."""
    if not is_cacheable_tool(tool):
        return None
    key = cache_key(tool, args)
    value = await redis_client.client.get(key)
    return cast(str | None, value)


async def set_cached(
    redis_client: RedisClient,
    tool: str,
    args: dict[str, Any],
    response: str,
    ttl: int = DEFAULT_CACHE_TTL,
) -> None:
    """Store MCP response with TTL; no-op for write tools."""
    if not is_cacheable_tool(tool):
        return
    key = cache_key(tool, args)
    await redis_client.client.set(key, response, ex=ttl)
