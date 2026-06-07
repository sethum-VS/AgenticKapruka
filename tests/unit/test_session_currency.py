"""Tests for session currency Redis storage and HTMX header selector."""

from __future__ import annotations

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.main import create_app
from app.templating import _create_templates, render_currency_selector
from lib.chat.session import SESSION_COOKIE_NAME, verify_signed_session_cookie
from lib.redis.client import RedisClient
from lib.redis.session import (
    DEFAULT_CURRENCY,
    get_session_currency,
    session_currency_key,
    set_session_currency,
)


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


def test_session_currency_key_format() -> None:
    assert session_currency_key("abc123") == "session:abc123:currency"


@pytest.mark.asyncio
async def test_get_session_currency_defaults_to_lkr(redis_client: RedisClient) -> None:
    assert await get_session_currency(redis_client, "new-session") == DEFAULT_CURRENCY


@pytest.mark.asyncio
async def test_set_and_get_session_currency(redis_client: RedisClient) -> None:
    session_id = "sess-usd-1"
    stored = await set_session_currency(redis_client, session_id, "usd")
    assert stored == "USD"
    assert await get_session_currency(redis_client, session_id) == "USD"
    raw = await redis_client.client.get(session_currency_key(session_id))
    assert raw == "USD"


@pytest.mark.asyncio
async def test_set_session_currency_rejects_unknown_code(redis_client: RedisClient) -> None:
    with pytest.raises(ValueError, match="currency must be one of"):
        await set_session_currency(redis_client, "sess-bad", "JPY")


def test_currency_selector_renders_htmx_post_with_options() -> None:
    html = render_currency_selector(currency="GBP")

    assert 'data-testid="header-currency"' in html
    assert 'name="currency"' in html
    assert 'hx-post="/session/currency"' in html
    assert 'hx-swap="none"' in html
    assert 'hx-trigger="change"' in html
    assert '<option value="LKR">LKR</option>' in html
    assert '<option value="USD">USD</option>' in html
    assert '<option value="GBP" selected>GBP</option>' in html
    assert '<option value="EUR">EUR</option>' in html


@pytest.fixture
def session_app(monkeypatch: pytest.MonkeyPatch, redis_client: RedisClient):
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    application = create_app()
    application.state.redis = redis_client
    return application


@pytest.mark.asyncio
async def test_post_session_currency_persists_in_redis(session_app) -> None:
    transport = ASGITransport(app=session_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/session/currency",
            data={"currency": "USD"},
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 204
    cookie_header = response.headers.get("set-cookie", "")
    assert "ak_session=" in cookie_header
    session_cookie = cookie_header.split("ak_session=", maxsplit=1)[1].split(";", maxsplit=1)[0]

    thread_id = verify_signed_session_cookie(session_cookie)
    assert thread_id is not None
    assert await get_session_currency(session_app.state.redis, thread_id) == "USD"


@pytest.mark.asyncio
async def test_currency_persists_across_page_refresh(session_app) -> None:
    transport = ASGITransport(app=session_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        post = await client.post(
            "/session/currency",
            data={"currency": "EUR"},
            headers={"HX-Request": "true"},
        )
        cookie_header = post.headers.get("set-cookie", "")
        session_cookie = cookie_header.split("ak_session=", maxsplit=1)[1].split(";", maxsplit=1)[0]

        first = await client.get(
            "/chat",
            headers={"Cookie": f"{SESSION_COOKIE_NAME}={session_cookie}"},
        )
        second = await client.get(
            "/chat",
            headers={"Cookie": f"{SESSION_COOKIE_NAME}={session_cookie}"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert '<option value="EUR" selected>EUR</option>' in first.text
    assert '<option value="EUR" selected>EUR</option>' in second.text
    assert '<option value="LKR" selected>LKR</option>' not in first.text


@pytest.mark.asyncio
async def test_post_session_currency_rejects_invalid_code(session_app) -> None:
    transport = ASGITransport(app=session_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/session/currency",
            data={"currency": "JPY"},
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 422
