"""Tests for chat dependency helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.runnables import RunnableConfig
from starlette.requests import Request

from lib.chat.deps import client_ip_from_request, resolve_turn_state


def _make_request(
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    client_host: str = "198.51.100.10",
) -> Request:
    scope: dict[str, object] = {
        "type": "http",
        "method": "POST",
        "path": "/chat/stream",
        "headers": headers or [],
        "query_string": b"",
        "client": (client_host, 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_client_ip_ignores_x_forwarded_for_from_untrusted_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    request = _make_request(
        headers=[(b"x-forwarded-for", b"1.2.3.4, 5.6.7.8")],
        client_host="198.51.100.10",
    )

    assert client_ip_from_request(request) == "198.51.100.10"


def test_client_ip_honors_x_forwarded_for_behind_local_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    request = _make_request(
        headers=[(b"x-forwarded-for", b"1.2.3.4, 5.6.7.8")],
        client_host="127.0.0.1",
    )

    assert client_ip_from_request(request) == "1.2.3.4"


@pytest.mark.asyncio
async def test_resolve_turn_state_uses_checkpoint_on_follow_up() -> None:
    graph = MagicMock()
    graph.aget_state = AsyncMock(return_value=MagicMock(values={"messages": []}))
    config: RunnableConfig = {"configurable": {"thread_id": "thread-follow-up"}}

    state = await resolve_turn_state(
        graph,
        message="second turn",
        session_id="thread-follow-up",
        zep_thread_id="thread-follow-up",
        config=config,
    )

    assert state["messages"][-1].content == "second turn"


@pytest.mark.asyncio
async def test_resolve_turn_state_seeds_currency_on_first_turn() -> None:
    graph = MagicMock()
    graph.aget_state = AsyncMock(return_value=MagicMock(values={}))
    config: RunnableConfig = {"configurable": {"thread_id": "thread-new"}}

    state = await resolve_turn_state(
        graph,
        message="first turn",
        session_id="thread-new",
        zep_thread_id="thread-new",
        config=config,
        currency="USD",
    )

    assert state["currency"] == "USD"


@pytest.mark.asyncio
async def test_resolve_turn_state_refreshes_currency_on_follow_up() -> None:
    graph = MagicMock()
    graph.aget_state = AsyncMock(
        return_value=MagicMock(values={"messages": [], "currency": "LKR"}),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "thread-follow-up"}}

    state = await resolve_turn_state(
        graph,
        message="second turn",
        session_id="thread-follow-up",
        zep_thread_id="thread-follow-up",
        config=config,
        currency="USD",
    )

    assert state["currency"] == "USD"
