"""End-to-end integration tests for POST /chat/stream."""

from __future__ import annotations

from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry
from tests.helpers.mock_genai import build_mock_genai_client
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.main import create_app
from graphs.shopping_graph import ShoppingGraphDeps
from lib.chat.session import SESSION_COOKIE_NAME, verify_signed_session_cookie
from lib.kapruka.service import KaprukaService
from lib.kapruka.types import CategoryRef, Money, ProductResult, SearchProductsOutput
from lib.redis.client import RedisClient

_MESSAGE = "show me birthday cakes"

_SEARCH_OUTPUT = SearchProductsOutput(
    results=[
        ProductResult(
            id="cake00ka002034",
            name="Chocolate Fudge Birthday Cake",
            summary="Rich chocolate layers.",
            price=Money(amount=4500.0, currency="LKR"),
            compare_at_price=None,
            in_stock=True,
            stock_level="high",
            image_url="https://example.com/cake.jpg",
            category=CategoryRef(id="cat_cakes", name="Birthday", slug="birthday"),
            rating=None,
            ships_internationally=False,
            url="https://www.kapruka.com/cake",
        ),
    ],
    next_cursor=None,
    applied_filters={"q": _MESSAGE, "limit": 10, "in_stock_only": False},
)


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    """Skip RediSearch index creation; fakeredis lacks FT._LIST."""
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


def _mock_kapruka_service() -> AsyncMock:
    mock_service = AsyncMock(spec=KaprukaService)
    mock_service.search_products.return_value = _SEARCH_OUTPUT
    return mock_service


@pytest.fixture
def chat_flow_env(monkeypatch: pytest.MonkeyPatch) -> RedisClient:
    """App env with fakeredis and mocked Kapruka + Gemini for full graph chat stream."""
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    redis_client = RedisClient("redis://localhost:6379/0", client=fake)

    deps = ShoppingGraphDeps(
        kapruka_service=_mock_kapruka_service(),
        client_ip="127.0.0.1",
        genai_client=build_mock_genai_client(
            search_query=_MESSAGE,
            assistant_message="Here are some birthday cake options from Kapruka.",
        ),
        zep_client=None,
    )

    async def mock_build_deps(request: object, redis: RedisClient) -> ShoppingGraphDeps:
        return deps

    monkeypatch.setattr("app.routes.chat.build_shopping_graph_deps", mock_build_deps)
    monkeypatch.setattr(AsyncRedisSaver, "asetup", _fakeredis_asetup)
    return redis_client


@pytest.mark.asyncio
async def test_chat_stream_end_to_end_assistant_message(chat_flow_env: RedisClient) -> None:
    """POST /chat/stream runs shopping graph; SSE contains assistant role markup."""
    application = create_app()
    application.state.redis = chat_flow_env
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            data={"message": _MESSAGE},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert "event: message\n" in body
    assert _MESSAGE in body
    assert 'aria-label="Assistant message"' in body
    assert 'role="assistant"' in body or 'role="article"' in body
    assert "birthday cake" in body.lower()
    assert 'data-testid="product-carousel"' in body
    assert "Chocolate Fudge Birthday Cake" in body
    assert 'data-product-id="cake00ka002034"' in body
    assert "Something went wrong" not in body
    assert 'role="alert"' not in body


@pytest.mark.asyncio
async def test_chat_stream_search_uses_session_currency_after_update(
    chat_flow_env: RedisClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session currency from Redis is passed to Kapruka search_products on chat stream."""
    mock_service = _mock_kapruka_service()
    deps = ShoppingGraphDeps(
        kapruka_service=mock_service,
        client_ip="127.0.0.1",
        genai_client=build_mock_genai_client(
            search_query=_MESSAGE,
            assistant_message="Here are some birthday cake options from Kapruka.",
        ),
        zep_client=None,
    )

    async def mock_build_deps(request: object, redis: RedisClient) -> ShoppingGraphDeps:
        return deps

    monkeypatch.setattr("app.routes.chat.build_shopping_graph_deps", mock_build_deps)

    application = create_app()
    application.state.redis = chat_flow_env
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        currency_response = await client.post(
            "/session/currency",
            data={"currency": "USD"},
            headers={"HX-Request": "true"},
        )
        cookie_header = currency_response.headers.get("set-cookie", "")
        session_cookie = cookie_header.split("ak_session=", maxsplit=1)[1].split(";", maxsplit=1)[0]

        response = await client.post(
            "/chat/stream",
            data={"message": _MESSAGE},
            headers={"Cookie": f"{SESSION_COOKIE_NAME}={session_cookie}"},
        )

    assert response.status_code == 200
    thread_id = verify_signed_session_cookie(session_cookie)
    assert thread_id is not None

    mock_service.search_products.assert_awaited_once()
    call_kwargs = mock_service.search_products.await_args.kwargs
    assert call_kwargs["currency"] == "USD"
