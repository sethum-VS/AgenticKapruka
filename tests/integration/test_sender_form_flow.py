"""Integration tests for sender form HTMX validation endpoint."""

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
def sender_form_app(monkeypatch: pytest.MonkeyPatch, redis_client: RedisClient):
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    application = create_app()
    application.state.redis = redis_client
    return application


@pytest.mark.asyncio
async def test_invalid_sender_returns_oob_error_without_losing_form_state(
    sender_form_app,
) -> None:
    """Empty name returns OOB error partial; anonymous checkbox state is preserved."""
    transport = ASGITransport(app=sender_form_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/checkout/validate-sender",
            data={
                "name": "",
                "anonymous": "true",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'data-testid="sender-name-error"' in response.text
    assert 'hx-swap-oob="innerHTML"' in response.text
    assert "checked" in response.text
    assert 'data-testid="sender-form-valid"' not in response.text


@pytest.mark.asyncio
async def test_valid_sender_form_returns_success_marker(sender_form_app) -> None:
    """Valid sender fields re-render form with success status and anonymous state."""
    transport = ASGITransport(app=sender_form_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/checkout/validate-sender",
            data={
                "name": "Ada Lovelace",
                "anonymous": "true",
            },
        )

    assert response.status_code == 200
    assert 'data-testid="sender-form-valid"' in response.text
    assert 'value="Ada Lovelace"' in response.text
    assert "checked" in response.text


@pytest.mark.asyncio
async def test_valid_sender_without_anonymous_checkbox(sender_form_app) -> None:
    """Unchecked anonymous checkbox omits field and parses as false."""
    transport = ASGITransport(app=sender_form_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/checkout/validate-sender",
            data={"name": "Charles Babbage"},
        )

    assert response.status_code == 200
    assert 'data-testid="sender-form-valid"' in response.text
    assert "anonymous: false" in response.text
