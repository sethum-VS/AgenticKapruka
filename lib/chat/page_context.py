"""Shared template context for server-rendered pages."""

from __future__ import annotations

from starlette.requests import Request

from app.templating import SUPPORTED_CURRENCY_CODES
from lib.chat.session import SESSION_COOKIE_NAME, verify_signed_session_cookie
from lib.redis.client import RedisClient
from lib.redis.session import DEFAULT_CURRENCY, get_session_currency


async def resolve_page_currency(request: Request, redis_client: RedisClient) -> str:
    """Load session currency from Redis when a valid session cookie is present."""
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    if existing:
        thread_id = verify_signed_session_cookie(existing)
        if thread_id:
            return await get_session_currency(redis_client, thread_id)
    return DEFAULT_CURRENCY


def currency_template_context(currency: str) -> dict[str, object]:
    """Template variables for header currency selector partials."""
    return {
        "currency": currency,
        "supported_currencies": SUPPORTED_CURRENCY_CODES,
    }
