"""Browser session preferences stored in Redis."""

from __future__ import annotations

from typing import Final, cast

from lib.redis.client import RedisClient
from lib.utils.currency import SUPPORTED_CURRENCIES

DEFAULT_CURRENCY: Final = "LKR"
SESSION_TTL_SECONDS: Final = 7 * 24 * 60 * 60  # 7 days, aligned with Zep session mapping


def session_currency_key(session_id: str) -> str:
    """Redis key for the shopper's preferred display currency."""
    return f"session:{session_id}:currency"


async def get_session_currency(redis_client: RedisClient, session_id: str) -> str:
    """Return stored currency for session_id, defaulting to LKR."""
    raw = cast(str | None, await redis_client.client.get(session_currency_key(session_id)))
    if raw is None:
        return DEFAULT_CURRENCY
    code = raw.upper()
    if code in SUPPORTED_CURRENCIES:
        return code
    return DEFAULT_CURRENCY


async def set_session_currency(
    redis_client: RedisClient,
    session_id: str,
    currency: str,
) -> str:
    """Persist currency for session_id; returns normalized uppercase code."""
    code = currency.strip().upper()
    if code not in SUPPORTED_CURRENCIES:
        msg = f"currency must be one of: {', '.join(sorted(SUPPORTED_CURRENCIES))}"
        raise ValueError(msg)
    await redis_client.client.set(
        session_currency_key(session_id),
        code,
        ex=SESSION_TTL_SECONDS,
    )
    return code
