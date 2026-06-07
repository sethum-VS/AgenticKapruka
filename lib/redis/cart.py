"""Server-side shopping cart stored in Redis hashes."""

from __future__ import annotations

import json
from typing import Final, cast

from pydantic import BaseModel, Field, field_validator

from lib.redis.client import RedisClient
from lib.redis.session import SESSION_TTL_SECONDS
from lib.utils.currency import SUPPORTED_CURRENCIES

MAX_CART_ITEMS: Final = 30
MIN_QUANTITY: Final = 1
MAX_QUANTITY: Final = 99


class CartLimitExceeded(Exception):
    """Raised when a cart already holds the maximum number of distinct line items."""

    def __init__(self, session_id: str, *, limit: int = MAX_CART_ITEMS) -> None:
        self.session_id = session_id
        self.limit = limit
        super().__init__(f"Cart for session {session_id} cannot exceed {limit} distinct items")


class CartItemNotFound(Exception):
    """Raised when an operation targets a product_id that is not in the cart."""

    def __init__(self, session_id: str, product_id: str) -> None:
        self.session_id = session_id
        self.product_id = product_id
        super().__init__(f"Product {product_id} not in cart for session {session_id}")


class StoredCartItem(BaseModel):
    """Cart line item with a price snapshot at add time."""

    product_id: str = Field(..., min_length=3, max_length=80)
    quantity: int = Field(default=1, ge=MIN_QUANTITY, le=MAX_QUANTITY)
    icing_text: str | None = Field(default=None, max_length=120)
    name: str = Field(..., min_length=1, max_length=200)
    price_amount: float = Field(..., ge=0)
    price_currency: str = "LKR"

    @field_validator("price_currency")
    @classmethod
    def validate_price_currency(cls, value: str) -> str:
        code = value.strip().upper()
        if code not in SUPPORTED_CURRENCIES:
            msg = f"price_currency must be one of: {', '.join(sorted(SUPPORTED_CURRENCIES))}"
            raise ValueError(msg)
        return code


def cart_key(session_id: str) -> str:
    """Redis hash key for the session cart."""
    return f"cart:{session_id}"


def _validate_quantity(quantity: int) -> int:
    if quantity < MIN_QUANTITY or quantity > MAX_QUANTITY:
        msg = f"quantity must be between {MIN_QUANTITY} and {MAX_QUANTITY}"
        raise ValueError(msg)
    return quantity


def _deserialize_item(raw: str) -> StoredCartItem:
    return StoredCartItem.model_validate(json.loads(raw))


async def _refresh_ttl(redis_client: RedisClient, session_id: str) -> None:
    key = cart_key(session_id)
    exists = cast(int, await redis_client.client.exists(key))
    if exists:
        await redis_client.client.expire(key, SESSION_TTL_SECONDS)


async def get_cart(redis_client: RedisClient, session_id: str) -> list[StoredCartItem]:
    """Return all cart line items for session_id (empty list when no cart)."""
    raw_items = cast(
        dict[str, str],
        await redis_client.client.hgetall(cart_key(session_id)),  # type: ignore[misc]
    )
    return [_deserialize_item(value) for value in raw_items.values()]


async def add_item(
    redis_client: RedisClient,
    session_id: str,
    *,
    product_id: str,
    name: str,
    price_amount: float,
    price_currency: str = "LKR",
    quantity: int = 1,
    icing_text: str | None = None,
) -> StoredCartItem:
    """Add or merge a line item; raises CartLimitExceeded when adding a 31st distinct product."""
    qty = _validate_quantity(quantity)
    key = cart_key(session_id)
    existing_raw = cast(
        str | None,
        await redis_client.client.hget(key, product_id),  # type: ignore[misc]
    )

    if existing_raw is not None:
        item = _deserialize_item(existing_raw)
        merged_qty = item.quantity + qty
        if merged_qty > MAX_QUANTITY:
            msg = f"quantity must be between {MIN_QUANTITY} and {MAX_QUANTITY}"
            raise ValueError(msg)
        item.quantity = merged_qty
        if icing_text is not None:
            item.icing_text = icing_text
    else:
        count = cast(int, await redis_client.client.hlen(key))  # type: ignore[misc]
        if count >= MAX_CART_ITEMS:
            raise CartLimitExceeded(session_id)
        item = StoredCartItem(
            product_id=product_id,
            quantity=qty,
            icing_text=icing_text,
            name=name,
            price_amount=price_amount,
            price_currency=price_currency,
        )

    await redis_client.client.hset(key, product_id, item.model_dump_json())  # type: ignore[misc]
    await _refresh_ttl(redis_client, session_id)
    return item


async def remove_item(
    redis_client: RedisClient,
    session_id: str,
    product_id: str,
) -> bool:
    """Remove product_id from the cart; returns True when an item was removed."""
    removed = cast(
        int,
        await redis_client.client.hdel(cart_key(session_id), product_id),  # type: ignore[misc]
    )
    if removed:
        await _refresh_ttl(redis_client, session_id)
    return bool(removed)


async def update_quantity(
    redis_client: RedisClient,
    session_id: str,
    product_id: str,
    quantity: int,
) -> StoredCartItem:
    """Set quantity for an existing line item."""
    qty = _validate_quantity(quantity)
    key = cart_key(session_id)
    existing_raw = cast(
        str | None,
        await redis_client.client.hget(key, product_id),  # type: ignore[misc]
    )
    if existing_raw is None:
        raise CartItemNotFound(session_id, product_id)

    item = _deserialize_item(existing_raw)
    item.quantity = qty
    await redis_client.client.hset(key, product_id, item.model_dump_json())  # type: ignore[misc]
    await _refresh_ttl(redis_client, session_id)
    return item


async def clear_cart(redis_client: RedisClient, session_id: str) -> None:
    """Delete all items for session_id."""
    await redis_client.client.delete(cart_key(session_id))
