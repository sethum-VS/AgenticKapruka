"""Sliding-window per-IP rate limiting backed by Redis sorted sets."""

from __future__ import annotations

import math
import time
import uuid
from typing import Any, Final, cast

from redis.asyncio import Redis

from lib.redis.client import RedisClient

CREATE_ORDER_TOOL: Final = "kapruka_create_order"

GLOBAL_WINDOW_SECONDS: Final = 60
GLOBAL_MAX_REQUESTS: Final = 60

CREATE_ORDER_WINDOW_SECONDS: Final = 3600
CREATE_ORDER_MAX_REQUESTS: Final = 30

_SLIDING_WINDOW_SCRIPT: Final = """
local key = KEYS[1]
local window_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window_ms)
local count = redis.call('ZCARD', key)

if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after_ms = window_ms
    if oldest[2] then
        retry_after_ms = tonumber(oldest[2]) + window_ms - now_ms
        if retry_after_ms < 1000 then
            retry_after_ms = 1000
        end
    end
    return {0, retry_after_ms}
end

redis.call('ZADD', key, now_ms, member)
redis.call('PEXPIRE', key, window_ms)
return {1, 0}
"""


class RateLimitExceeded(Exception):
    """Raised when a client exceeds a configured sliding-window limit."""

    def __init__(
        self,
        retry_after_seconds: int,
        *,
        limit_type: str,
        ip: str,
        tool_name: str,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        self.limit_type = limit_type
        self.ip = ip
        self.tool_name = tool_name
        super().__init__(
            f"Rate limit exceeded for {ip} ({limit_type}); "
            f"retry after {retry_after_seconds}s (tool={tool_name})"
        )


def rate_limit_key(ip: str, suffix: str) -> str:
    """Build Redis key for a client IP and limit bucket."""
    safe_ip = ip.replace(":", "_")
    return f"rate:{safe_ip}:{suffix}"


def retry_after_header(exc: RateLimitExceeded) -> dict[str, str]:
    """Map a rate-limit exception to HTTP Retry-After headers."""
    return {"Retry-After": str(exc.retry_after_seconds)}


async def check_rate_limit(
    redis_client: RedisClient,
    ip: str,
    tool_name: str,
) -> None:
    """Enforce global and create-order sliding windows; raise on exceed."""
    redis = redis_client.client

    allowed, retry_after = await _check_window(
        redis,
        rate_limit_key(ip, "global"),
        window_seconds=GLOBAL_WINDOW_SECONDS,
        limit=GLOBAL_MAX_REQUESTS,
    )
    if not allowed:
        raise RateLimitExceeded(
            retry_after,
            limit_type="global",
            ip=ip,
            tool_name=tool_name,
        )

    if tool_name == CREATE_ORDER_TOOL:
        allowed, retry_after = await _check_window(
            redis,
            rate_limit_key(ip, "create_order"),
            window_seconds=CREATE_ORDER_WINDOW_SECONDS,
            limit=CREATE_ORDER_MAX_REQUESTS,
        )
        if not allowed:
            raise RateLimitExceeded(
                retry_after,
                limit_type="create_order",
                ip=ip,
                tool_name=tool_name,
            )


async def _check_window(
    redis: Redis,
    key: str,
    *,
    window_seconds: int,
    limit: int,
) -> tuple[bool, int]:
    """Return (allowed, retry_after_seconds) for one sliding window bucket."""
    now_ms = int(time.time() * 1000)
    window_ms = window_seconds * 1000
    member = f"{now_ms}:{uuid.uuid4().hex}"

    result = cast(
        list[Any],
        await redis.eval(  # type: ignore[misc]
            _SLIDING_WINDOW_SCRIPT,
            1,
            key,
            str(window_ms),
            str(limit),
            str(now_ms),
            member,
        ),
    )
    allowed = int(result[0]) == 1
    retry_after_ms = int(result[1])
    retry_after_seconds = max(1, math.ceil(retry_after_ms / 1000))
    return allowed, retry_after_seconds
