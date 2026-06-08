"""Aggregate dependency health into a single readiness response."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.tools.list_categories import list_categories
from lib.neo4j.client import Neo4jClient
from lib.redis.client import RedisClient
from lib.zep.client import ZepClient

logger = logging.getLogger(__name__)

ServiceStatus = Literal["up", "down"]
OverallStatus = Literal["healthy", "degraded"]


class ServiceHealth(BaseModel):
    """Per-dependency probe result."""

    status: ServiceStatus


class ServicesHealth(BaseModel):
    """Health of each critical backend dependency."""

    redis: ServiceHealth
    neo4j: ServiceHealth
    zep: ServiceHealth
    mcp: ServiceHealth


class AggregatedHealthResponse(BaseModel):
    """JSON body for GET /health."""

    status: OverallStatus
    services: ServicesHealth = Field(
        description="Individual dependency probe results",
    )


async def _check_redis(client: RedisClient | None) -> ServiceHealth:
    if client is None:
        return ServiceHealth(status="down")
    try:
        ok = await client.ping()
    except Exception:
        logger.exception("Redis health probe failed")
        return ServiceHealth(status="down")
    return ServiceHealth(status="up" if ok else "down")


async def _check_neo4j(client: Neo4jClient | None) -> ServiceHealth:
    if client is None:
        return ServiceHealth(status="down")
    try:
        ok = await client.health_check()
    except Exception:
        logger.exception("Neo4j health probe failed")
        return ServiceHealth(status="down")
    return ServiceHealth(status="up" if ok else "down")


async def _check_zep(client: ZepClient | None) -> ServiceHealth:
    if client is None:
        return ServiceHealth(status="down")
    try:
        ok = await client.health_check()
    except Exception:
        logger.exception("Zep health probe failed")
        return ServiceHealth(status="down")
    return ServiceHealth(status="up" if ok else "down")


async def _check_mcp(client: MCPHttpClient | None) -> ServiceHealth:
    if client is None:
        return ServiceHealth(status="down")
    try:
        await list_categories(client, depth=1)
    except Exception:
        logger.exception("MCP health probe failed")
        return ServiceHealth(status="down")
    return ServiceHealth(status="up")


async def aggregate_health(app: FastAPI) -> tuple[AggregatedHealthResponse, int]:
    """Probe Redis, Neo4j, Zep, and Kapruka MCP; return body and HTTP status."""
    redis_client: RedisClient | None = getattr(app.state, "redis", None)
    neo4j_client: Neo4jClient | None = getattr(app.state, "neo4j", None)
    zep_client: ZepClient | None = getattr(app.state, "zep", None)
    mcp_client: MCPHttpClient | None = getattr(app.state, "mcp_client", None)

    services = ServicesHealth(
        redis=await _check_redis(redis_client),
        neo4j=await _check_neo4j(neo4j_client),
        zep=await _check_zep(zep_client),
        mcp=await _check_mcp(mcp_client),
    )

    all_up = all(
        svc.status == "up"
        for svc in (
            services.redis,
            services.neo4j,
            services.zep,
            services.mcp,
        )
    )
    overall: OverallStatus = "healthy" if all_up else "degraded"
    status_code = 200 if all_up else 503
    return AggregatedHealthResponse(status=overall, services=services), status_code
