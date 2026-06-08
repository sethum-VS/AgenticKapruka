"""Aggregated health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from lib.health.aggregator import AggregatedHealthResponse, aggregate_health

router = APIRouter()


@router.get("/health", response_model=AggregatedHealthResponse)
async def health(request: Request) -> JSONResponse:
    """Readiness probe: Redis, Neo4j, Zep, and Kapruka MCP."""
    body, status_code = await aggregate_health(request.app)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
    )
