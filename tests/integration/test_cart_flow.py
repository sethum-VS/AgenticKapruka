"""Integration tests for cart HTMX partial swap routes."""

from __future__ import annotations

from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.main import create_app
from lib.chat.session import SESSION_COOKIE_NAME, verify_signed_session_cookie
from lib.kapruka.service import KaprukaService
from lib.kapruka.types import (
    CategoryRef,
    GetProductOutput,
    Money,
    ProductAttributes,
    ProductShipping,
)
from lib.redis.cart import get_cart
from lib.redis.client import RedisClient

_PRODUCT_ID = "cake00ka002034"
_PRODUCT_NAME = "Chocolate Fudge Birthday Cake"

_GET_PRODUCT_OUTPUT = GetProductOutput(
    id=_PRODUCT_ID,
    name=_PRODUCT_NAME,
    description="Rich chocolate layers.",
    summary="Rich chocolate layers.",
    price=Money(amount=4500.0, currency="LKR"),
    compare_at_price=None,
    in_stock=True,
    stock_level="high",
    category=CategoryRef(id="cat_cakes", name="Birthday", slug="birthday"),
    variants=[],
    images=[],
    attributes=ProductAttributes(),
    shipping=ProductShipping(
        ships_from="Colombo",
        ships_internationally=False,
        restricted_countries=[],
    ),
    rating=None,
    url="https://www.kapruka.com/cake",
)


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.fixture
def cart_app(monkeypatch: pytest.MonkeyPatch, redis_client: RedisClient):
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    application = create_app()
    application.state.redis = redis_client

    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.get_product.return_value = _GET_PRODUCT_OUTPUT
    application.state.kapruka_service = mock_service

    return application


def _session_cookie_from_response(response) -> str:
    cookie_header = response.headers.get("set-cookie", "")
    return cookie_header.split("ak_session=", maxsplit=1)[1].split(";", maxsplit=1)[0]


@pytest.mark.asyncio
async def test_get_cart_panel_returns_current_cart_partial(cart_app) -> None:
    transport = ASGITransport(app=cart_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        add = await client.post(
            "/cart/add",
            data={"product_id": _PRODUCT_ID},
            headers={"HX-Request": "true"},
        )
        session_cookie = _session_cookie_from_response(add)
        panel = await client.get(
            "/cart/panel",
            headers={"HX-Request": "true", "Cookie": f"{SESSION_COOKIE_NAME}={session_cookie}"},
        )

    assert panel.status_code == 200
    assert _PRODUCT_NAME in panel.text
    assert 'data-testid="cart-line-item"' in panel.text
    assert 'data-item-count="1"' in panel.text


@pytest.mark.asyncio
async def test_post_cart_add_returns_html_with_product_name(cart_app) -> None:
    transport = ASGITransport(app=cart_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/cart/add",
            data={"product_id": _PRODUCT_ID},
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 200
    assert _PRODUCT_NAME in response.text
    assert 'id="cart-panel"' in response.text
    assert 'data-testid="cart-panel"' in response.text
    assert 'data-testid="cart-line-item"' in response.text

    session_cookie = _session_cookie_from_response(response)
    thread_id = verify_signed_session_cookie(session_cookie)
    assert thread_id is not None
    cart = await get_cart(cart_app.state.redis, thread_id)
    assert len(cart) == 1
    assert cart[0].name == _PRODUCT_NAME


@pytest.mark.asyncio
async def test_cart_update_and_remove_return_refreshed_partial(cart_app) -> None:
    transport = ASGITransport(app=cart_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        add = await client.post(
            "/cart/add",
            data={"product_id": _PRODUCT_ID},
            headers={"HX-Request": "true"},
        )
        session_cookie = _session_cookie_from_response(add)

        update = await client.post(
            "/cart/update",
            data={"product_id": _PRODUCT_ID, "quantity": 3},
            headers={"HX-Request": "true", "Cookie": f"{SESSION_COOKIE_NAME}={session_cookie}"},
        )
        remove = await client.post(
            "/cart/remove",
            data={"product_id": _PRODUCT_ID},
            headers={"HX-Request": "true", "Cookie": f"{SESSION_COOKIE_NAME}={session_cookie}"},
        )

    assert update.status_code == 200
    assert 'data-item-count="3"' in update.text
    assert remove.status_code == 200
    assert 'data-testid="cart-empty"' in remove.text

    thread_id = verify_signed_session_cookie(session_cookie)
    assert thread_id is not None
    assert await get_cart(cart_app.state.redis, thread_id) == []
