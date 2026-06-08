"""Integration tests for recipient form HTMX validation endpoint."""

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
def recipient_form_app(monkeypatch: pytest.MonkeyPatch, redis_client: RedisClient):
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    application = create_app()
    application.state.redis = redis_client
    return application


@pytest.mark.asyncio
async def test_invalid_phone_returns_oob_error_without_losing_form_state(
    recipient_form_app,
) -> None:
    """Invalid phone returns OOB error partial; name value remains in the form."""
    transport = ASGITransport(app=recipient_form_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/checkout/validate-recipient",
            data={
                "name": "Ada Lovelace",
                "phone": "12345",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'data-testid="recipient-phone-error"' in response.text
    assert 'hx-swap-oob="innerHTML"' in response.text
    assert 'value="Ada Lovelace"' in response.text
    assert 'data-testid="recipient-form-valid"' not in response.text


@pytest.mark.asyncio
async def test_valid_recipient_form_returns_success_marker(recipient_form_app) -> None:
    """Valid recipient fields re-render form with success status."""
    transport = ASGITransport(app=recipient_form_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/checkout/validate-recipient",
            data={
                "name": "Ada Lovelace",
                "phone": "+94771234567",
            },
        )

    assert response.status_code == 200
    assert 'data-testid="recipient-form-valid"' in response.text
    assert 'value="Ada Lovelace"' in response.text
    assert 'value="+94771234567"' in response.text
