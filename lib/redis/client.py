"""Async Redis client wrapper with connection pooling and reconnect logic."""

from __future__ import annotations

import logging
from typing import Any, cast
from urllib.parse import urlparse

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONNECTIONS = 10
_DEFAULT_SOCKET_CONNECT_TIMEOUT = 5.0
_DEFAULT_HEALTH_CHECK_INTERVAL = 30


def _redacted_redis_endpoint(url: str) -> str:
    """Return host:port for logging without credentials from the Redis URL."""
    parsed = urlparse(url)
    host = parsed.hostname or "unknown"
    if parsed.port is not None:
        return f"{host}:{parsed.port}"
    return host


class RedisClient:
    """Wrap redis-py async client with pooling, health checks, and reconnect."""

    def __init__(
        self,
        url: str,
        *,
        client: Redis | None = None,
        max_connections: int = _DEFAULT_MAX_CONNECTIONS,
        socket_connect_timeout: float = _DEFAULT_SOCKET_CONNECT_TIMEOUT,
        socket_keepalive: bool = True,
        health_check_interval: int = _DEFAULT_HEALTH_CHECK_INTERVAL,
    ) -> None:
        self._url = url
        self._pool_kwargs: dict[str, Any] = {
            "encoding": "utf-8",
            "decode_responses": True,
            "max_connections": max_connections,
            "socket_connect_timeout": socket_connect_timeout,
            "socket_keepalive": socket_keepalive,
            "health_check_interval": health_check_interval,
            "retry_on_timeout": True,
        }
        self._client = client

    @classmethod
    async def connect(
        cls,
        url: str,
        *,
        max_connections: int = _DEFAULT_MAX_CONNECTIONS,
        socket_connect_timeout: float = _DEFAULT_SOCKET_CONNECT_TIMEOUT,
        socket_keepalive: bool = True,
        health_check_interval: int = _DEFAULT_HEALTH_CHECK_INTERVAL,
    ) -> RedisClient:
        """Create a connected client from a Redis URL (Memorystore or local)."""
        instance = cls(
            url,
            max_connections=max_connections,
            socket_connect_timeout=socket_connect_timeout,
            socket_keepalive=socket_keepalive,
            health_check_interval=health_check_interval,
        )
        instance._client = cast(
            Redis,
            aioredis.from_url(url, **instance._pool_kwargs),  # type: ignore[no-untyped-call]
        )
        return instance

    @property
    def client(self) -> Redis:
        """Underlying redis.asyncio client for cache, rate limit, and cart modules."""
        if self._client is None:
            msg = "RedisClient is not connected; call connect() first"
            raise RuntimeError(msg)
        return self._client

    async def ping(self) -> bool:
        """Return True when Redis responds to PING; reconnect once on failure."""
        for attempt in range(2):
            try:
                redis_client = self._require_client()
                result = await redis_client.ping()
                return bool(result)
            except (RedisConnectionError, RedisTimeoutError, OSError) as exc:
                logger.warning("Redis ping failed (attempt %s): %s", attempt + 1, exc)
                if attempt == 0:
                    await self._reconnect()
                    continue
                return False
        return False

    async def close(self) -> None:
        """Close the connection pool; safe to call multiple times."""
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None
        logger.debug("Redis connection pool closed")

    def _require_client(self) -> Redis:
        if self._client is None:
            self._client = cast(
                Redis,
                aioredis.from_url(self._url, **self._pool_kwargs),  # type: ignore[no-untyped-call]
            )
        return self._client

    async def _reconnect(self) -> None:
        """Drop the current pool and open a fresh connection."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except (RedisConnectionError, RedisTimeoutError, OSError) as exc:
                logger.debug("Error closing Redis pool during reconnect: %s", exc)
        self._client = cast(
            Redis,
            aioredis.from_url(self._url, **self._pool_kwargs),  # type: ignore[no-untyped-call]
        )
        logger.info("Redis client reconnected to %s", _redacted_redis_endpoint(self._url))
