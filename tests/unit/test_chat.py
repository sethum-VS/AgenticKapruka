"""Tests for chat page routes and templates."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from app.main import create_app
from app.templating import _create_templates, get_templates


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
        {"title": "Chat — AgenticKapruka"},
    )

    html = response.body.decode()

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="chat-messages"' in html
    assert 'id="chat-empty-state"' in html
    assert "What would you like to send today?" in html
    assert "Birthday cake for mom in Colombo" in html
    assert "htmx.org" in html
    assert 'href="/static/css/app.css"' in html
    assert 'id="chat-form"' in html
    assert 'hx-post="/chat/stream"' in html
    assert 'hx-ext="sse"' in html
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
    assert "/static/js/chat-helpers.js" in html


@pytest.mark.asyncio
async def test_chat_stream_post_returns_user_bubble_html() -> None:
    """POST /chat/stream returns HTMX-swappable user message HTML."""
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            data={"message": "Birthday cake for mom"},
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "Birthday cake for mom" in html
    assert 'hx-swap-oob="delete"' in html
    assert 'id="chat-empty-state"' in html
    assert 'aria-label="Your message"' in html


@pytest.mark.asyncio
async def test_chat_stream_rejects_empty_message() -> None:
    """POST /chat/stream rejects blank messages."""
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            data={"message": "   "},
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_index_returns_200_html_with_empty_state() -> None:
    """GET /chat renders HTML with empty state visible."""
    application = create_app()
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
    assert 'x-data="chatHelpers()"' in html
    assert "/static/js/chat-helpers.js" in html
