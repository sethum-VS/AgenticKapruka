"""E2E application factory: fakeredis, mock MCP, and mock Gemini for smoke tests."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import fakeredis.aioredis
from evals.ragas_eval import build_eval_genai_client
from fastapi import FastAPI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry
from tests.fixtures.mcp_mock import MockMCPHttpClient
from tests.unit.test_settings import _VALID_ENV

from app.config import get_settings
from app.main import create_app
from app.routes import chat as chat_route
from graphs.shopping_graph import ShoppingGraphDeps
from lib.chat.deps import client_ip_from_request
from lib.kapruka.service import KaprukaService
from lib.redis.client import RedisClient

E2E_PORT = 8080
_mcp_client: MockMCPHttpClient | None = None
_redis_client: RedisClient | None = None


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    """Skip RediSearch index creation; fakeredis lacks FT._LIST."""
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


def _apply_e2e_env() -> None:
    get_settings.cache_clear()
    for key, value in _VALID_ENV.items():
        os.environ[key] = value
    os.environ["APP_ENV"] = "development"


def get_e2e_mcp_client() -> MockMCPHttpClient:
    """Return the shared mock MCP client wired into the E2E app."""
    if _mcp_client is None:
        msg = "E2E app not initialized; call create_e2e_app() first"
        raise RuntimeError(msg)
    return _mcp_client


def create_e2e_app() -> FastAPI:
    """Build FastAPI app with in-memory Redis, mock MCP, and mock Gemini."""
    global _mcp_client, _redis_client

    _apply_e2e_env()
    AsyncRedisSaver.asetup = _fakeredis_asetup  # type: ignore[method-assign]

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    _redis_client = RedisClient("redis://localhost:6379/0", client=fake)
    _mcp_client = MockMCPHttpClient()
    kapruka_service = KaprukaService(_redis_client, _mcp_client)

    application = create_app()

    @asynccontextmanager
    async def e2e_lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.redis = _redis_client
        app.state.mcp_client = _mcp_client
        app.state.kapruka_service = kapruka_service
        app.state.neo4j = None
        app.state.zep = None
        yield

    application.router.lifespan_context = e2e_lifespan

    async def mock_build_deps(request: Any, redis: RedisClient) -> ShoppingGraphDeps:
        return ShoppingGraphDeps(
            kapruka_service=kapruka_service,
            client_ip=client_ip_from_request(request),
            genai_client=build_eval_genai_client("discovery"),
            zep_client=None,
            redis_client=redis,
        )

    chat_route.build_shopping_graph_deps = mock_build_deps

    @application.get("/e2e/mcp-calls", include_in_schema=False)
    async def e2e_mcp_calls() -> dict[str, list[str]]:
        """Expose mock MCP call log for smoke-test assertions (E2E only)."""
        return {"tools": list(_mcp_client.call_log)}

    return application
