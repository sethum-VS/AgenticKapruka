"""Tests for signed browser session cookies."""

from __future__ import annotations

import pytest
from starlette.requests import Request
from tests.unit.test_settings import _VALID_ENV, _apply_env

from app.config import get_settings
from lib.chat.session import (
    SESSION_COOKIE_NAME,
    resolve_chat_thread_id,
    verify_signed_session_cookie,
)


def _request_with_cookie(value: str | None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if value is not None:
        headers.append((b"cookie", f"{SESSION_COOKIE_NAME}={value}".encode()))
    scope: dict[str, object] = {
        "type": "http",
        "method": "POST",
        "path": "/chat/stream",
        "headers": headers,
        "query_string": b"",
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


@pytest.fixture(autouse=True)
def settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)
    yield
    get_settings.cache_clear()


def test_resolve_chat_thread_id_mints_signed_cookie_when_missing() -> None:
    thread_id, cookie = resolve_chat_thread_id(_request_with_cookie(None))

    assert cookie is not None
    assert verify_signed_session_cookie(cookie) == thread_id


def test_resolve_chat_thread_id_reuses_valid_signed_cookie() -> None:
    request = _request_with_cookie(None)
    thread_id, cookie = resolve_chat_thread_id(request)
    assert cookie is not None

    follow_up = _request_with_cookie(cookie)
    reused_thread_id, new_cookie = resolve_chat_thread_id(follow_up)

    assert reused_thread_id == thread_id
    assert new_cookie is None


def test_resolve_chat_thread_id_rejects_forged_opaque_cookie() -> None:
    thread_id, cookie = resolve_chat_thread_id(_request_with_cookie("attacker-controlled-id"))

    assert cookie is not None
    assert thread_id != "attacker-controlled-id"
    assert verify_signed_session_cookie(cookie) == thread_id
