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
async def test_chat_stub_returns_200() -> None:
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/chat")
    assert response.status_code == 200
    assert response.json() == {"status": "stub", "route": "chat"}


@pytest.mark.asyncio
async def test_health_stub_returns_200() -> None:
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
