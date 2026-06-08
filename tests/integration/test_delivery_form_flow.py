"""Integration tests for delivery form HTMX validation endpoint."""

from __future__ import annotations

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.main import create_app
from lib.redis.client import RedisClient


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
def delivery_form_app(monkeypatch: pytest.MonkeyPatch, redis_client: RedisClient):
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    application = create_app()
    application.state.redis = redis_client
    return application


@pytest.mark.asyncio
async def test_invalid_address_returns_oob_error_without_losing_form_state(
    delivery_form_app,
) -> None:
    """Short address returns OOB error partial; other field values remain in the form."""
    transport = ASGITransport(app=delivery_form_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/checkout/validate-delivery",
            data={
                "address": "ab",
                "city": "Colombo 03",
                "location_type": "house",
                "date": "2026-06-10",
                "instructions": "Ring the bell",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'data-testid="delivery-address-error"' in response.text
    assert 'hx-swap-oob="innerHTML"' in response.text
    assert 'value="Colombo 03"' in response.text
    assert "Ring the bell" in response.text
    assert 'data-testid="delivery-form-valid"' not in response.text


@pytest.mark.asyncio
async def test_valid_delivery_form_returns_success_marker(delivery_form_app) -> None:
    """Valid delivery fields re-render form with success status."""
    transport = ASGITransport(app=delivery_form_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/checkout/validate-delivery",
            data={
                "address": "42 Lotus Road",
                "city": "Colombo 03",
                "location_type": "apartment",
                "date": "2026-06-10",
                "instructions": "",
            },
        )

    assert response.status_code == 200
    assert 'data-testid="delivery-form-valid"' in response.text
    assert 'value="42 Lotus Road"' in response.text
    assert '<option value="apartment" selected>' in response.text
