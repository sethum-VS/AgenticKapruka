"""Unit tests for cart price refresh on currency switch."""

from __future__ import annotations

from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.main import create_app
from lib.cart.pricing import refresh_cart_prices_for_currency
from lib.chat.session import SESSION_COOKIE_NAME, verify_signed_session_cookie
from lib.kapruka.service import KaprukaService
from lib.kapruka.types import (
    CategoryRef,
    GetProductOutput,
    Money,
    ProductAttributes,
    ProductShipping,
)
from lib.redis.cart import add_item, get_cart
from lib.redis.client import RedisClient
from lib.redis.session import get_session_currency

_PRODUCT_ID = "cake00ka002034"
_PRODUCT_NAME = "Chocolate Fudge Birthday Cake"

_LKR_PRODUCT = GetProductOutput(
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

_USD_PRODUCT = _LKR_PRODUCT.model_copy(
    update={"price": Money(amount=15.0, currency="USD")},
)


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.mark.asyncio
async def test_refresh_cart_prices_for_currency_updates_mismatched_lines(
    redis_client: RedisClient,
) -> None:
    session_id = "sess-refresh-1"
    await add_item(
        redis_client,
        session_id,
        product_id=_PRODUCT_ID,
        name=_PRODUCT_NAME,
        price_amount=4500.0,
        price_currency="LKR",
    )

    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.get_product.return_value = _USD_PRODUCT

    refreshed = await refresh_cart_prices_for_currency(
        redis_client,
        session_id,
        currency="USD",
        kapruka_service=mock_service,
        client_ip="127.0.0.1",
    )

    assert len(refreshed) == 1
    assert refreshed[0].price_amount == 15.0
    assert refreshed[0].price_currency == "USD"
    stored = await get_cart(redis_client, session_id)
    assert stored[0].price_currency == "USD"
    mock_service.get_product.assert_awaited_once_with(
        "127.0.0.1",
        product_id=_PRODUCT_ID,
        currency="USD",
    )


@pytest.mark.asyncio
async def test_refresh_cart_prices_skips_when_currency_already_matches(
    redis_client: RedisClient,
) -> None:
    session_id = "sess-refresh-2"
    await add_item(
        redis_client,
        session_id,
        product_id=_PRODUCT_ID,
        name=_PRODUCT_NAME,
        price_amount=4500.0,
        price_currency="LKR",
    )

    mock_service = AsyncMock(spec=KaprukaService)

    refreshed = await refresh_cart_prices_for_currency(
        redis_client,
        session_id,
        currency="LKR",
        kapruka_service=mock_service,
        client_ip="127.0.0.1",
    )

    assert refreshed[0].price_amount == 4500.0
    mock_service.get_product.assert_not_called()


@pytest.fixture
def session_app(monkeypatch: pytest.MonkeyPatch, redis_client: RedisClient):
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    application = create_app()
    application.state.redis = redis_client

    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.get_product.return_value = _USD_PRODUCT
    application.state.kapruka_service = mock_service

    return application


@pytest.mark.asyncio
async def test_post_session_currency_returns_oob_cart_with_refreshed_prices(
    session_app,
    redis_client: RedisClient,
) -> None:
    transport = ASGITransport(app=session_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        bootstrap = await client.post(
            "/session/currency",
            data={"currency": "LKR"},
            headers={"HX-Request": "true"},
        )
        session_cookie = bootstrap.headers.get("set-cookie", "").split("ak_session=", maxsplit=1)[1]
        session_cookie = session_cookie.split(";", maxsplit=1)[0]
        thread_id = verify_signed_session_cookie(session_cookie)
        assert thread_id is not None

        await add_item(
            redis_client,
            thread_id,
            product_id=_PRODUCT_ID,
            name=_PRODUCT_NAME,
            price_amount=4500.0,
            price_currency="LKR",
        )

        session_app.state.kapruka_service.get_product.return_value = _USD_PRODUCT

        response = await client.post(
            "/session/currency",
            data={"currency": "USD"},
            headers={
                "HX-Request": "true",
                "Cookie": f"{SESSION_COOKIE_NAME}={session_cookie}",
            },
        )

    assert response.status_code == 200
    assert 'id="cart-panel" hx-swap-oob="outerHTML"' in response.text
    assert "$15.00" in response.text
    assert await get_session_currency(redis_client, thread_id) == "USD"
    cart = await get_cart(redis_client, thread_id)
    assert cart[0].price_currency == "USD"
    assert cart[0].price_amount == 15.0
