"""Integration tests for delivery city HTMX autocomplete partial."""

from __future__ import annotations

from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.main import create_app
from lib.kapruka.service import KaprukaService
from lib.redis.client import RedisClient

_COLOMBO_CITIES = [
    "Colombo 01",
    "Colombo 02",
    "Colombo 03",
    "Colombo 04",
    "Colombo 05",
]


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
def delivery_city_app(monkeypatch: pytest.MonkeyPatch, redis_client: RedisClient):
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    application = create_app()
    application.state.redis = redis_client

    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.list_delivery_cities.return_value = _COLOMBO_CITIES
    application.state.kapruka_service = mock_service

    return application


@pytest.mark.asyncio
async def test_typing_col_returns_colombo_suggestion_html(delivery_city_app) -> None:
    """GET /partials/delivery-cities?q=Col returns Colombo suggestion li items."""
    transport = ASGITransport(app=delivery_city_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/partials/delivery-cities", params={"q": "Col"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'data-testid="delivery-city-suggestion"' in response.text
    assert "Colombo 01" in response.text
    assert "Colombo 03" in response.text

    mock_service = delivery_city_app.state.kapruka_service
    mock_service.list_delivery_cities.assert_awaited_once()
    call_kwargs = mock_service.list_delivery_cities.await_args.kwargs
    assert call_kwargs["query"] == "Col"
    assert call_kwargs["limit"] == 10


@pytest.mark.asyncio
async def test_short_query_returns_empty_suggestions(delivery_city_app) -> None:
    """Queries shorter than two characters skip Kapruka lookup."""
    transport = ASGITransport(app=delivery_city_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/partials/delivery-cities", params={"q": "C"})

    assert response.status_code == 200
    assert response.text == ""

    mock_service = delivery_city_app.state.kapruka_service
    mock_service.list_delivery_cities.assert_not_awaited()
