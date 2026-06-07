"""Server-side checkout sub-graph state stored per browser session."""

from __future__ import annotations

import json
from typing import Any, Final, cast

from graphs.checkout_state import CheckoutState
from lib.redis.client import RedisClient
from lib.redis.session import SESSION_TTL_SECONDS

_CHECKOUT_FIELDS: Final = (
    "current_step",
    "step_valid",
    "delivery_city",
    "delivery_address",
    "delivery_location_type",
    "delivery_date",
    "delivery_instructions",
    "recipient_name",
    "recipient_phone",
    "sender_name",
    "sender_anonymous",
    "gift_message",
    "currency",
)


def checkout_state_key(session_id: str) -> str:
    """Redis key for persisted checkout sub-graph fields."""
    return f"session:{session_id}:checkout"


def _serialize_checkout_state(state: CheckoutState) -> str:
    payload: dict[str, Any] = {}
    for field in _CHECKOUT_FIELDS:
        if field in state and state[field] is not None:
            payload[field] = state[field]
    return json.dumps(payload)


def _deserialize_checkout_state(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        return {}
    return data


async def get_checkout_session(
    redis_client: RedisClient,
    session_id: str,
) -> dict[str, Any]:
    """Load persisted checkout fields for session_id, or {} when missing."""
    raw = cast(str | None, await redis_client.client.get(checkout_state_key(session_id)))
    if raw is None:
        return {}
    return _deserialize_checkout_state(raw)


async def save_checkout_session(
    redis_client: RedisClient,
    session_id: str,
    state: CheckoutState,
) -> None:
    """Persist checkout sub-graph fields (cart lines are stored separately)."""
    await redis_client.client.set(
        checkout_state_key(session_id),
        _serialize_checkout_state(state),
        ex=SESSION_TTL_SECONDS,
    )


async def clear_checkout_session(redis_client: RedisClient, session_id: str) -> None:
    """Remove persisted checkout state after a successful order."""
    await redis_client.client.delete(checkout_state_key(session_id))
