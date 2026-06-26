"""Tests for aggregated GET /health."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from lib.health.aggregator import AggregatedHealthResponse, aggregate_health
from lib.kapruka.mcp_client import MCPHttpClient
from lib.neo4j.client import Neo4jClient
from lib.redis.client import RedisClient
from lib.zep.client import ZepClient


def _healthy_services() -> dict[str, dict[str, str]]:
    return {
        "redis": {"status": "up"},
        "neo4j": {"status": "up"},
        "neo4j_graphrag": {"status": "up"},
        "zep": {"status": "up"},
        "mcp": {"status": "up"},
    }


def _patch_graphrag_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lib.health.aggregator.has_category_embeddings",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "lib.health.aggregator.has_category_vector_index",
        AsyncMock(return_value=True),
    )


@pytest.fixture
def all_healthy_clients() -> dict[str, Any]:
    """Mock clients that pass every health probe."""
    redis = MagicMock(spec=RedisClient)
    redis.ping = AsyncMock(return_value=True)

    neo4j = MagicMock(spec=Neo4jClient)
    neo4j.health_check = AsyncMock(return_value=True)

    zep = MagicMock(spec=ZepClient)
    zep.health_check = AsyncMock(return_value=True)

    mcp = MagicMock(spec=MCPHttpClient)
    mcp.ping = AsyncMock(return_value=True)

    return {
        "redis": redis,
        "neo4j": neo4j,
        "zep": zep,
        "mcp_client": mcp,
    }


@pytest.mark.asyncio
async def test_aggregate_health_healthy_when_all_services_up(
    all_healthy_clients: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All probes up yields healthy status and HTTP 200."""
    _patch_graphrag_healthy(monkeypatch)

    app = MagicMock()
    app.state = MagicMock()
    for key, client in all_healthy_clients.items():
        setattr(app.state, key, client)

    body, status_code = await aggregate_health(app)

    assert status_code == 200
    assert body.status == "healthy"
    assert body.services.model_dump() == _healthy_services()


@pytest.mark.asyncio
async def test_aggregate_health_degraded_when_redis_down(
    all_healthy_clients: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any failed probe yields degraded status and HTTP 503."""
    _patch_graphrag_healthy(monkeypatch)
    all_healthy_clients["redis"].ping = AsyncMock(return_value=False)

    app = MagicMock()
    app.state = MagicMock()
    for key, client in all_healthy_clients.items():
        setattr(app.state, key, client)

    body, status_code = await aggregate_health(app)

    assert status_code == 503
    assert body.status == "degraded"
    assert body.services.redis.status == "down"
    assert body.services.neo4j.status == "up"


@pytest.mark.asyncio
async def test_aggregate_health_degraded_when_mcp_ping_fails(
    all_healthy_clients: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP lightweight ping failure marks mcp down."""
    _patch_graphrag_healthy(monkeypatch)
    all_healthy_clients["mcp_client"].ping = AsyncMock(
        side_effect=RuntimeError("mcp unavailable"),
    )

    app = MagicMock()
    app.state = MagicMock()
    for key, client in all_healthy_clients.items():
        setattr(app.state, key, client)

    body, status_code = await aggregate_health(app)

    assert status_code == 503
    assert body.status == "degraded"
    assert body.services.mcp.status == "down"


@pytest.mark.asyncio
async def test_health_endpoint_returns_schema_with_mocked_services(
    all_healthy_clients: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /health returns AggregatedHealthResponse JSON with mocked app.state."""
    _patch_graphrag_healthy(monkeypatch)

    application = create_app()
    for key, client in all_healthy_clients.items():
        setattr(application.state, key, client)

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    parsed = AggregatedHealthResponse.model_validate(payload)
    assert parsed.status == "healthy"
    assert parsed.services.model_dump() == _healthy_services()


@pytest.mark.asyncio
async def test_health_endpoint_returns_503_when_degraded(
    all_healthy_clients: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /health uses 503 when any critical dependency is down."""
    _patch_graphrag_healthy(monkeypatch)
    all_healthy_clients["zep"].health_check = AsyncMock(return_value=False)

    application = create_app()
    for key, client in all_healthy_clients.items():
        setattr(application.state, key, client)

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["services"]["zep"]["status"] == "down"


@pytest.mark.asyncio
async def test_aggregate_health_degraded_when_graphrag_index_missing(
    all_healthy_clients: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neo4j connectivity up but missing vector index marks neo4j_graphrag down."""
    monkeypatch.setattr(
        "lib.health.aggregator.has_category_embeddings",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "lib.health.aggregator.has_category_vector_index",
        AsyncMock(return_value=False),
    )

    app = MagicMock()
    app.state = MagicMock()
    for key, client in all_healthy_clients.items():
        setattr(app.state, key, client)

    body, status_code = await aggregate_health(app)

    assert status_code == 503
    assert body.status == "degraded"
    assert body.services.neo4j.status == "up"
    assert body.services.neo4j_graphrag.status == "down"


@pytest.mark.asyncio
async def test_aggregate_health_degraded_when_neo4j_client_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Neo4j client marks both neo4j and neo4j_graphrag down."""
    redis = MagicMock(spec=RedisClient)
    redis.ping = AsyncMock(return_value=True)
    zep = MagicMock(spec=ZepClient)
    zep.health_check = AsyncMock(return_value=True)
    mcp = MagicMock(spec=MCPHttpClient)
    mcp.ping = AsyncMock(return_value=True)

    app = MagicMock()
    app.state = MagicMock()
    app.state.redis = redis
    app.state.neo4j = None
    app.state.zep = zep
    app.state.mcp_client = mcp

    body, status_code = await aggregate_health(app)

    assert status_code == 503
    assert body.services.neo4j.status == "down"
    assert body.services.neo4j_graphrag.status == "down"
