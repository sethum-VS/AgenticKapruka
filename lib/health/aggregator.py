"""Aggregate dependency health into a single readiness response."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from lib.kapruka.mcp_client import MCPHttpClient
from lib.neo4j.client import Neo4jClient
from lib.neo4j.embed_ontology import has_category_embeddings
from lib.neo4j.vector_search import has_category_vector_index
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
    neo4j_graphrag: ServiceHealth
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


async def _check_neo4j_graphrag(client: Neo4jClient | None) -> ServiceHealth:
    if client is None:
        return ServiceHealth(status="down")
    try:
        if not await client.health_check():
            return ServiceHealth(status="down")
        has_embeddings = await has_category_embeddings(client)
        has_index = await has_category_vector_index(client)
    except Exception:
        logger.exception("Neo4j GraphRAG health probe failed")
        return ServiceHealth(status="down")
    ready = has_embeddings and has_index
    return ServiceHealth(status="up" if ready else "down")


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
        ok = await client.ping()
    except Exception:
        logger.exception("MCP health probe failed")
        return ServiceHealth(status="down")
    return ServiceHealth(status="up" if ok else "down")


async def aggregate_health(app: FastAPI) -> tuple[AggregatedHealthResponse, int]:
    """Probe Redis, Neo4j, Zep, and Kapruka MCP; return body and HTTP status."""
    redis_client: RedisClient | None = getattr(app.state, "redis", None)
    neo4j_client: Neo4jClient | None = getattr(app.state, "neo4j", None)
    zep_client: ZepClient | None = getattr(app.state, "zep", None)
    mcp_client: MCPHttpClient | None = getattr(app.state, "mcp_client", None)

    (
        redis_health,
        neo4j_health,
        neo4j_graphrag_health,
        zep_health,
        mcp_health,
    ) = await asyncio.gather(
        _check_redis(redis_client),
        _check_neo4j(neo4j_client),
        _check_neo4j_graphrag(neo4j_client),
        _check_zep(zep_client),
        _check_mcp(mcp_client),
    )

    services = ServicesHealth(
        redis=redis_health,
        neo4j=neo4j_health,
        neo4j_graphrag=neo4j_graphrag_health,
        zep=zep_health,
        mcp=mcp_health,
    )

    all_up = all(
        svc.status == "up"
        for svc in (
            services.redis,
            services.neo4j,
            services.neo4j_graphrag,
            services.zep,
            services.mcp,
        )
    )
    overall: OverallStatus = "healthy" if all_up else "degraded"
    status_code = 200 if all_up else 503
    return AggregatedHealthResponse(status=overall, services=services), status_code
