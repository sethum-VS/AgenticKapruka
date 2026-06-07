"""Browser session identifiers for chat continuity."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from typing import Final, Literal, TypedDict

from starlette.requests import Request

from app.config import get_settings

SESSION_COOKIE_NAME: Final = "ak_session"
_COOKIE_MAX_AGE: Final = 7 * 24 * 60 * 60


def cookie_secure() -> bool:
    """Use Secure cookies when APP_ENV is production."""
    return os.getenv("APP_ENV", "development").lower() == "production"


class SessionCookieParams(TypedDict):
    httponly: bool
    samesite: Literal["lax"]
    max_age: int
    secure: bool


def cookie_params() -> SessionCookieParams:
    """Standard Set-Cookie kwargs for the browser session."""
    return {
        "httponly": True,
        "samesite": "lax",
        "max_age": _COOKIE_MAX_AGE,
        "secure": cookie_secure(),
    }


def _sign_thread_id(thread_id: str) -> str:
    settings = get_settings()
    payload = base64.urlsafe_b64encode(thread_id.encode()).decode().rstrip("=")
    sig = hmac.new(
        settings.session_secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()[:32]
    return f"{payload}.{sig}"


def verify_signed_session_cookie(cookie_value: str) -> str | None:
    """Return thread_id when the signed ak_session cookie is valid."""
    value = cookie_value.strip()
    if not value or "." not in value:
        return None
    payload, sig = value.rsplit(".", 1)
    settings = get_settings()
    expected = hmac.new(
        settings.session_secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()[:32]
    if not hmac.compare_digest(sig, expected):
        return None
    pad = "=" * (-len(payload) % 4)
    try:
        return base64.urlsafe_b64decode(payload + pad).decode()
    except (ValueError, UnicodeDecodeError):
        return None


def resolve_chat_thread_id(request: Request) -> tuple[str, str | None]:
    """Return LangGraph thread_id and an optional new signed cookie value.

    Rejects client-supplied opaque IDs — only server-signed cookies are trusted.
    """
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    if existing:
        thread_id = verify_signed_session_cookie(existing)
        if thread_id:
            return thread_id, None
    thread_id = secrets.token_urlsafe(32)
    return thread_id, _sign_thread_id(thread_id)
