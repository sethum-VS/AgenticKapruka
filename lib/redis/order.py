"""Pending Kapruka checkout order metadata stored per browser session."""

from __future__ import annotations

from typing import Final, NamedTuple, cast

from lib.redis.client import RedisClient
from lib.redis.session import SESSION_TTL_SECONDS


class PendingOrder(NamedTuple):
    """order_ref and expires_at from kapruka_create_order."""

    order_ref: str
    expires_at: str


def session_order_ref_key(session_id: str) -> str:
    """Redis key for the pre-payment Kapruka order reference."""
    return f"session:{session_id}:order_ref"


def session_order_expires_at_key(session_id: str) -> str:
    """Redis key for the checkout link expiry timestamp (ISO 8601)."""
    return f"session:{session_id}:order_expires_at"


_ORDER_TTL_SECONDS: Final = SESSION_TTL_SECONDS


async def store_pending_order(
    redis_client: RedisClient,
    session_id: str,
    *,
    order_ref: str,
    expires_at: str,
) -> None:
    """Persist order_ref and expires_at for the session (PRD-071 payment CTA)."""
    ref_key = session_order_ref_key(session_id)
    exp_key = session_order_expires_at_key(session_id)
    await redis_client.client.set(ref_key, order_ref, ex=_ORDER_TTL_SECONDS)
    await redis_client.client.set(exp_key, expires_at, ex=_ORDER_TTL_SECONDS)


async def get_pending_order(
    redis_client: RedisClient,
    session_id: str,
) -> PendingOrder | None:
    """Return stored order_ref and expires_at when both keys are present."""
    ref_key = session_order_ref_key(session_id)
    exp_key = session_order_expires_at_key(session_id)
    order_ref = cast(str | None, await redis_client.client.get(ref_key))
    expires_at = cast(str | None, await redis_client.client.get(exp_key))
    if not order_ref or not expires_at:
        return None
    return PendingOrder(order_ref=order_ref, expires_at=expires_at)
