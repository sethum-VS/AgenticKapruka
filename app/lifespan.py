"""Application lifespan: startup/shutdown for Redis, Neo4j, Zep, and MCP clients."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.config import get_settings
from lib.kapruka.mcp_client import MCPHttpClient
from lib.neo4j.client import Neo4jClient
from lib.redis.client import RedisClient
from lib.zep.client import ZepClient

logger = logging.getLogger(__name__)


async def _close_client(client: Any, name: str) -> None:
    """Close a service client if it exposes async close()."""
    if client is None:
        return
    close = getattr(client, "close", None)
    if close is None:
        return
    await close()
    logger.info("%s client closed", name)


async def _connect_optional[T](
    name: str,
    connect: Callable[[], Awaitable[T]],
) -> T | None:
    """Connect a dependency; log and return None when startup probe fails."""
    try:
        client = await connect()
    except Exception:
        logger.exception("%s connection failed during startup", name)
        return None
    logger.info("%s client connected", name)
    return client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Connect external services on startup; release resources on shutdown."""
    settings = get_settings()
    app.state.redis = await _connect_optional(
        "Redis",
        lambda: RedisClient.connect(settings.redis_url),
    )
    app.state.neo4j = await _connect_optional(
        "Neo4j",
        lambda: Neo4jClient.connect(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        ),
    )
    app.state.zep = await _connect_optional(
        "Zep",
        lambda: ZepClient.connect(settings.zep_api_key),
    )
    app.state.mcp_client = await _connect_optional(
        "Kapruka MCP",
        lambda: MCPHttpClient.connect(settings.kapruka_mcp_url),
    )
    logger.info("Application startup complete")

    yield

    await _close_client(app.state.mcp_client, "Kapruka MCP")
    await _close_client(app.state.redis, "Redis")
    await _close_client(app.state.neo4j, "Neo4j")
    await _close_client(app.state.zep, "Zep")
    logger.info("Application shutdown complete")
