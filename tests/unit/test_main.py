"""Tests for FastAPI application factory."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.mark.asyncio
async def test_root_redirects_to_chat() -> None:
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/chat"


@pytest.mark.asyncio
async def test_health_returns_aggregated_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /health exposes status and per-service probes (mocked in test_health.py)."""
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr(
        "lib.health.aggregator.has_category_embeddings",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "lib.health.aggregator.has_category_vector_index",
        AsyncMock(return_value=True),
    )

    application = create_app()
    application.state.redis = MagicMock()
    application.state.redis.ping = AsyncMock(return_value=True)
    application.state.neo4j = MagicMock()
    application.state.neo4j.health_check = AsyncMock(return_value=True)
    application.state.zep = MagicMock()
    application.state.zep.health_check = AsyncMock(return_value=True)
    application.state.mcp_client = MagicMock()
    application.state.mcp_client.ping = AsyncMock(return_value=True)

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert set(payload["services"]) == {
        "redis",
        "neo4j",
        "neo4j_graphrag",
        "zep",
        "mcp",
    }
    assert all(svc["status"] == "up" for svc in payload["services"].values())


@pytest.mark.asyncio
async def test_static_css_returns_200_with_text_css() -> None:
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/css/app.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    assert response.text


@pytest.mark.asyncio
async def test_static_vendor_js_returns_200() -> None:
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/js/vendor/htmx.min.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert response.text
