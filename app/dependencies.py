"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import HTTPException, Request, status

from lib.redis.client import RedisClient


async def get_redis(request: Request) -> AsyncGenerator[RedisClient, None]:
    """Yield the shared Redis client stored on application state during lifespan."""
    client: RedisClient | None = getattr(request.app.state, "redis", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is not available",
        )
    yield client
