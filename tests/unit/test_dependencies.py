"""Tests for FastAPI dependency injection."""

from __future__ import annotations

from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.dependencies import get_redis
from app.lifespan import lifespan
from lib.kapruka.mcp_client import MCPHttpClient
from lib.neo4j.client import Neo4jClient
from lib.redis.client import RedisClient
from lib.zep.client import ZepClient

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
async def test_lifespan_stores_service_clients_on_app_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifespan connects Redis, Neo4j, Zep, and MCP clients on app.state."""
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)

    captured: dict[str, object] = {}
    fake = fakeredis.aioredis.FakeRedis()

    @classmethod
    async def mock_redis_connect(cls, url: str, **kwargs: Any) -> RedisClient:
        client = RedisClient(url, client=fake)
        captured["redis"] = client
        return client

    @classmethod
    async def mock_neo4j_connect(
        cls,
        uri: str,
        user: str,
        password: str,
        **kwargs: Any,
    ) -> Neo4jClient:
        driver = MagicMock()
        driver.close = AsyncMock()
        client = Neo4jClient(uri, user, password, driver=driver)
        captured["neo4j"] = client
        return client

    @classmethod
    async def mock_zep_connect(cls, api_key: str, **kwargs: Any) -> ZepClient:
        client = ZepClient(api_key, client=MagicMock())
        captured["zep"] = client
        return client

    @classmethod
    async def mock_mcp_connect(
        cls,
        url: str = "https://mcp.kapruka.com/mcp",
        **kwargs: Any,
    ) -> MCPHttpClient:
        client = MCPHttpClient(url)
        captured["mcp"] = client
        return client

    monkeypatch.setattr(RedisClient, "connect", mock_redis_connect)
    monkeypatch.setattr(Neo4jClient, "connect", mock_neo4j_connect)
    monkeypatch.setattr(ZepClient, "connect", mock_zep_connect)
    monkeypatch.setattr(MCPHttpClient, "connect", mock_mcp_connect)

    application = FastAPI()
    async with lifespan(application):
        assert application.state.redis is captured["redis"]
        assert application.state.neo4j is captured["neo4j"]
        assert application.state.zep is captured["zep"]
        assert application.state.mcp_client is captured["mcp"]
        assert await application.state.redis.ping() is True
