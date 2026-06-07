"""Browser session identifiers for chat continuity."""

from __future__ import annotations

import secrets
from typing import Final

from starlette.requests import Request

SESSION_COOKIE_NAME: Final = "ak_session"


def get_session_id(request: Request) -> str:
    """Return stable session id from cookie or generate a new one."""
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    if existing and existing.strip():
        return existing.strip()
    return secrets.token_urlsafe(32)
