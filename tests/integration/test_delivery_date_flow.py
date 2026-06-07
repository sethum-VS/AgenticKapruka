"""Integration tests for delivery date HTMX check-delivery endpoint."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.main import create_app
from lib.kapruka.service import KaprukaService
from lib.kapruka.types import CheckDeliveryOutput
from lib.redis.client import RedisClient

COLOMBO = ZoneInfo("Asia/Colombo")
_COLOMBO_TODAY = datetime(2026, 6, 8, 14, 0, tzinfo=COLOMBO)


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
def delivery_date_app(monkeypatch: pytest.MonkeyPatch, redis_client: RedisClient):
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    application = create_app()
    application.state.redis = redis_client

    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.check_delivery.return_value = CheckDeliveryOutput(
        city="Colombo 03",
        now="2026-06-08T14:00:00+05:30",
        checked_date="2026-06-10",
        available=True,
        rate=350.0,
        currency="LKR",
    )
    application.state.kapruka_service = mock_service

    return application


@pytest.mark.asyncio
async def test_past_date_rejected_with_error_partial(delivery_date_app) -> None:
    """POST with a past date returns user-friendly error partial without MCP call."""
    with patch("app.routes.checkout.colombo_today_iso", return_value="2026-06-08"):
        transport = ASGITransport(app=delivery_date_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/checkout/check-delivery",
                data={"city": "Colombo 03", "delivery_date": "2026-06-07"},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'data-testid="delivery-date-error"' in response.text
    assert "Date in the past" in response.text
    assert "2026-06-08" in response.text
    assert "Asia/Colombo" in response.text

    mock_service = delivery_date_app.state.kapruka_service
    mock_service.check_delivery.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_date_calls_check_delivery_and_returns_status(delivery_date_app) -> None:
    """POST with today-or-future date validates via Kapruka and returns status HTML."""
    with patch("app.routes.checkout.colombo_today_iso", return_value="2026-06-08"):
        transport = ASGITransport(app=delivery_date_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/checkout/check-delivery",
                data={"city": "Colombo 03", "delivery_date": "2026-06-10"},
            )

    assert response.status_code == 200
    assert 'data-testid="delivery-date-available"' in response.text
    assert "2026-06-10" in response.text

    mock_service = delivery_date_app.state.kapruka_service
    mock_service.check_delivery.assert_awaited_once()
    call_kwargs = mock_service.check_delivery.await_args.kwargs
    assert call_kwargs["city"] == "Colombo 03"
    assert call_kwargs["delivery_date"] == "2026-06-10"
