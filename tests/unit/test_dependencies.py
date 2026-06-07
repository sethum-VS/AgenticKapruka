"""Tests for FastAPI dependency injection."""

from __future__ import annotations

from typing import Annotated, Any

import fakeredis.aioredis
import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.dependencies import get_redis
from app.lifespan import lifespan
from lib.redis.client import RedisClient

RedisDep = Annotated[RedisClient, Depends(get_redis)]


@pytest.mark.asyncio
async def test_get_redis_dependency_resolves() -> None:
    """Route using Depends(get_redis) receives the shared app.state Redis client."""
    fake = fakeredis.aioredis.FakeRedis()
    redis_client = RedisClient("redis://localhost:6379/0", client=fake)

    application = FastAPI()
    application.state.redis = redis_client

    @application.get("/redis-ping")
    async def redis_ping(redis: RedisDep) -> dict[str, bool]:
        return {"ping": await redis.ping()}

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/redis-ping")

    assert response.status_code == 200
    assert response.json() == {"ping": True}


@pytest.mark.asyncio
async def test_get_redis_returns_503_when_unavailable() -> None:
    """Missing app.state.redis yields 503 from the dependency."""
    application = FastAPI()
    application.state.redis = None

    @application.get("/redis-ping")
    async def redis_ping(redis: RedisDep) -> dict[str, bool]:
        return {"ping": await redis.ping()}

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/redis-ping")

    assert response.status_code == 503
    assert response.json() == {"detail": "Redis is not available"}


@pytest.mark.asyncio
async def test_lifespan_stores_redis_on_app_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifespan connects RedisClient and assigns it to app.state.redis."""
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)

    captured: dict[str, RedisClient] = {}
    fake = fakeredis.aioredis.FakeRedis()

    @classmethod
    async def mock_connect(cls, url: str, **kwargs: Any) -> RedisClient:
        client = RedisClient(url, client=fake)
        captured["client"] = client
        return client

    monkeypatch.setattr(RedisClient, "connect", mock_connect)

    application = FastAPI()
    async with lifespan(application):
        assert "client" in captured
        assert application.state.redis is captured["client"]
        assert await application.state.redis.ping() is True
