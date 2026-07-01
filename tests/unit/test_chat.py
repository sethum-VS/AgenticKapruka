"""Tests for chat page routes and templates."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from app.main import create_app
from app.templating import SUPPORTED_CURRENCY_CODES, _create_templates, get_templates
from graphs.nodes.generate_response import render_assistant_html
from graphs.shopping_graph import ShoppingGraphDeps
from lib.chat.page_context import cart_template_context, currency_template_context
from lib.chat.session import SESSION_COOKIE_NAME, _sign_thread_id
from lib.redis.cart import add_item, get_cart
from lib.redis.client import RedisClient


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def _make_request() -> Request:
    scope: dict[str, object] = {
        "type": "http",
        "method": "GET",
        "path": "/chat",
        "headers": [],
        "query_string": b"",
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_chat_index_template_renders_empty_state() -> None:
    """chat/index.html extends base.html with message container and welcome state."""
    templates = get_templates()
    request = _make_request()
    response = templates.TemplateResponse(
        request,
        "chat/index.html",
        {
            "title": "Chat — AgenticKapruka",
            **currency_template_context("LKR"),
            **cart_template_context([]),
        },
    )

    html = response.body.decode()

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="chat-messages"' in html
    assert 'id="chat-empty-state"' in html
    assert "Welcome!" in html
    assert "personal gifting concierge" in html
    assert "Birthday cake for mom in Colombo" in html
    assert 'data-chat-suggestion="Birthday cake for mom in Colombo"' in html
    assert 'type="button"' in html
    assert "htmx.org" not in html
    assert "/static/js/vendor/htmx.min.js" in html
    assert 'href="/static/css/app.css"' in html
    assert 'id="chat-form"' in html
    assert 'hx-post="/chat/stream"' in html
    assert 'hx-ext="sse"' in html
    assert 'sse-connect="/chat/stream"' in html
    assert 'sse-swap="message,status,carousel,done"' in html
    assert 'id="chat-sse-listener"' in html
    assert 'hx-target="#chat-messages"' in html
    assert 'hx-swap="beforeend"' in html
    assert 'hx-trigger="submit"' in html
    assert 'hx-indicator="#chat-loading"' in html
    assert 'name="message"' in html
    assert 'id="chat-loading"' in html
    assert "htmx-indicator" in html
    assert 'x-data="chatHelpers()"' in html
    assert 'x-ref="messages"' in html
    assert 'x-ref="input"' in html
    assert "/static/js/chat-sse.js" in html
    assert 'defer src="/static/js/chat-helpers.js"' in html
    assert html.index('defer src="/static/js/chat-helpers.js"') < html.index(
        'defer src="/static/js/vendor/alpine.min.js"'
    )
    assert 'defer src="/static/js/lazy-image.js"' in html
    assert "/static/js/cart-drawer.js" in html
    assert 'data-testid="cart-drawer"' in html
    assert 'data-testid="cart-icon"' in html
    assert 'id="cart-panel"' in html
    assert 'data-testid="header-currency"' in html
    assert 'data-testid="new-chat-button"' in html
    assert 'hx-post="/chat/new"' in html
    assert "htmx.ajax('GET','/cart/panel'" in html
    assert 'hx-post="/session/currency"' in html
    assert 'hx-swap="none"' in html
    for code in SUPPORTED_CURRENCY_CODES:
        assert f'<option value="{code}"' in html


def _mock_streaming_graph() -> MagicMock:
    """Graph mock that emits a single generate_response astream update."""
    assistant_html = render_assistant_html("Here are some birthday cake options.")
    mock_graph = MagicMock()

    async def fake_astream(
        state: object,
        config: dict[str, Any],
        stream_mode: str | list[str] | None = None,
    ) -> Any:
        yield (
            "updates",
            {
                "generate_response": {
                    "response_html": assistant_html,
                    "assistant_message": "Here are some birthday cake options.",
                },
            },
        )

    mock_graph.astream = fake_astream
    mock_graph.aget_state = AsyncMock(return_value=MagicMock(values=None))
    return mock_graph


@pytest.fixture
def chat_stream_env(monkeypatch: pytest.MonkeyPatch) -> RedisClient:
    """App env with fakeredis and mocked LangGraph for /chat/stream tests."""
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    redis_client = RedisClient("redis://localhost:6379/0", client=fake)

    mock_graph = _mock_streaming_graph()

    async def mock_get_compiled_chat_graph(
        redis: RedisClient,
        *,
        deps: ShoppingGraphDeps | None = None,
    ) -> MagicMock:
        return mock_graph

    async def mock_build_deps(request: object, redis: RedisClient) -> ShoppingGraphDeps:
        return ShoppingGraphDeps(
            kapruka_service=AsyncMock(),
            client_ip="127.0.0.1",
            genai_client=MagicMock(),
            zep_client=None,
        )

    monkeypatch.setattr("app.routes.chat.get_compiled_chat_graph", mock_get_compiled_chat_graph)
    monkeypatch.setattr("app.routes.chat.build_shopping_graph_deps", mock_build_deps)
    return redis_client


@pytest.mark.asyncio
async def test_chat_stream_returns_sse_event_stream(chat_stream_env: RedisClient) -> None:
    """POST /chat/stream returns text/event-stream with valid SSE framing."""
    application = create_app()
    application.state.redis = chat_stream_env
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            data={"message": "Birthday cake for mom"},
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: message\n" in body
    assert "data:" in body
    assert body.count("\n\n") >= 2
    assert "Birthday cake for mom" in body
    assert 'hx-swap-oob="delete"' in body
    assert 'id="chat-empty-state"' in body
    assert 'aria-label="Your message"' in body
    assert "birthday cake options" in body.lower()
    assert "ak_session" in response.headers.get("set-cookie", "")


@pytest.mark.asyncio
async def test_chat_stream_reuses_session_cookie_on_follow_up(
    chat_stream_env: RedisClient,
) -> None:
    """Second POST with returned cookie does not emit another Set-Cookie."""
    application = create_app()
    application.state.redis = chat_stream_env
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/chat/stream",
            data={"message": "Hello"},
            headers={"HX-Request": "true"},
        )
        cookie_header = first.headers.get("set-cookie", "")
        assert "ak_session=" in cookie_header
        session_cookie = cookie_header.split("ak_session=", maxsplit=1)[1].split(";", maxsplit=1)[0]

        second = await client.post(
            "/chat/stream",
            data={"message": "Follow up"},
            headers={"HX-Request": "true", "Cookie": f"ak_session={session_cookie}"},
        )

    assert second.status_code == 200
    assert "set-cookie" not in second.headers


@pytest.mark.asyncio
async def test_chat_stream_setup_failure_surfaces_error(
    chat_stream_env: RedisClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph setup failures yield a user-visible SSE error fragment."""

    async def boom(*_args: object, **_kwargs: object) -> None:
        msg = "graph compile failed"
        raise RuntimeError(msg)

    monkeypatch.setattr("app.routes.chat.get_compiled_chat_graph", boom)
    application = create_app()
    application.state.redis = chat_stream_env
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            data={"message": "Hello"},
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 200
    assert 'role="alert"' in response.text
    assert "Something went wrong" in response.text


@pytest.mark.asyncio
async def test_chat_stream_rejects_empty_message(chat_stream_env: RedisClient) -> None:
    """POST /chat/stream rejects blank messages."""
    application = create_app()
    application.state.redis = chat_stream_env
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            data={"message": "   "},
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 422


@pytest.fixture
def chat_index_env(monkeypatch: pytest.MonkeyPatch) -> RedisClient:
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


@pytest.mark.asyncio
async def test_chat_index_returns_200_html_with_empty_state(chat_index_env: RedisClient) -> None:
    """GET /chat renders HTML with empty state visible."""
    application = create_app()
    application.state.redis = chat_index_env
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/chat")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert 'id="chat-messages"' in html
    assert 'id="chat-empty-state"' in html
    assert "Kapruka Gift Assistant" in html
    assert "Gift ideas under Rs. 5,000" in html
    assert 'id="chat-form"' in html
    assert 'hx-post="/chat/stream"' in html
    assert 'sse-connect="/chat/stream"' in html
    assert 'sse-swap="message,status,carousel,done"' in html
    assert 'x-data="chatHelpers()"' in html
    assert "/static/js/chat-sse.js" in html
    assert 'defer src="/static/js/chat-helpers.js"' in html
    assert html.index('defer src="/static/js/chat-helpers.js"') < html.index(
        'defer src="/static/js/vendor/alpine.min.js"'
    )
    assert 'defer src="/static/js/lazy-image.js"' in html
    assert 'data-testid="header-currency"' in html
    assert '<option value="LKR" selected>LKR</option>' in html


@pytest.mark.asyncio
async def test_chat_new_clears_cart_and_oob_empty_panel(chat_index_env: RedisClient) -> None:
    """POST /chat/new rotates session and clears Redis cart for the new thread."""
    old_thread = "thread-cart-clear-test"
    await add_item(
        chat_index_env,
        old_thread,
        product_id="cake00ka001827",
        name="Happy Birthday Symphony Ribbon Cake",
        price_amount=6500.0,
        price_currency="LKR",
        quantity=1,
    )
    application = create_app()
    application.state.redis = chat_index_env
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/new",
            headers={"HX-Request": "true"},
            cookies={SESSION_COOKIE_NAME: _sign_thread_id(old_thread)},
        )

    assert response.status_code == 200
    assert 'id="cart-panel" hx-swap-oob="outerHTML"' in response.text
    assert 'data-item-count="0"' in response.text
    set_cookie = response.headers.get("set-cookie", "")
    assert SESSION_COOKIE_NAME in set_cookie
    new_cookie = response.cookies[SESSION_COOKIE_NAME]
    from lib.chat.session import verify_signed_session_cookie

    new_thread = verify_signed_session_cookie(new_cookie)
    assert new_thread is not None
    assert new_thread != old_thread
    assert await get_cart(chat_index_env, old_thread) == []
    assert await get_cart(chat_index_env, new_thread) == []
