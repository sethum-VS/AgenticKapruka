"""Application lifespan: startup/shutdown for Redis, Neo4j, and Zep clients."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.config import get_settings
from lib.redis.client import RedisClient

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Connect external services on startup; release resources on shutdown."""
    settings = get_settings()
    app.state.redis = await RedisClient.connect(settings.redis_url)
    # Neo4j and Zep wired in PRD-012 and PRD-013.
    app.state.neo4j = None
    app.state.zep = None
    logger.info("Application startup complete")

    yield

    await _close_client(app.state.redis, "Redis")
    await _close_client(app.state.neo4j, "Neo4j")
    await _close_client(app.state.zep, "Zep")
    logger.info("Application shutdown complete")
