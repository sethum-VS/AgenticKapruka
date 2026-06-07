"""Unit tests for Redis-backed server-side cart state."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from lib.redis.cart import (
    MAX_CART_ITEMS,
    CartItemNotFound,
    CartLimitExceeded,
    StoredCartItem,
    add_item,
    cart_key,
    clear_cart,
    get_cart,
    remove_item,
    update_quantity,
)
from lib.redis.client import RedisClient


@pytest.fixture
def redis_client() -> RedisClient:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisClient("redis://localhost:6379/0", client=fake)


def _line(
    index: int,
    *,
    quantity: int = 1,
    icing_text: str | None = None,
) -> dict[str, object]:
    return {
        "product_id": f"prod{index:04d}",
        "name": f"Product {index}",
        "price_amount": float(1000 + index),
        "price_currency": "LKR",
        "quantity": quantity,
        "icing_text": icing_text,
    }


@pytest.mark.asyncio
async def test_cart_key_format() -> None:
    assert cart_key("sess-abc") == "cart:sess-abc"


@pytest.mark.asyncio
async def test_add_item_and_get_cart(redis_client: RedisClient) -> None:
    session_id = "sess-add-1"
    item = await add_item(
        redis_client,
        session_id,
        product_id="cake00ka002034",
        name="Chocolate Fudge Birthday Cake",
        price_amount=4500.0,
        price_currency="LKR",
        quantity=2,
        icing_text="Happy Birthday",
    )

    assert item == StoredCartItem(
        product_id="cake00ka002034",
        name="Chocolate Fudge Birthday Cake",
        price_amount=4500.0,
        price_currency="LKR",
        quantity=2,
        icing_text="Happy Birthday",
    )

    cart = await get_cart(redis_client, session_id)
    assert len(cart) == 1
    assert cart[0].product_id == "cake00ka002034"
    assert cart[0].quantity == 2


@pytest.mark.asyncio
async def test_add_item_merges_quantity_for_existing_product(redis_client: RedisClient) -> None:
    session_id = "sess-merge-1"
    await add_item(
        redis_client,
        session_id,
        product_id="cake00ka002034",
        name="Chocolate Cake",
        price_amount=4500.0,
        quantity=2,
    )
    await add_item(
        redis_client,
        session_id,
        product_id="cake00ka002034",
        name="Chocolate Cake",
        price_amount=4500.0,
        quantity=3,
        icing_text="Congrats",
    )

    cart = await get_cart(redis_client, session_id)
    assert len(cart) == 1
    assert cart[0].quantity == 5
    assert cart[0].icing_text == "Congrats"
    assert cart[0].price_amount == 4500.0


@pytest.mark.asyncio
async def test_adding_31st_item_raises_cart_limit_exceeded(redis_client: RedisClient) -> None:
    session_id = "sess-limit-1"
    for index in range(MAX_CART_ITEMS):
        await add_item(redis_client, session_id, **_line(index))

    cart = await get_cart(redis_client, session_id)
    assert len(cart) == MAX_CART_ITEMS

    with pytest.raises(CartLimitExceeded) as exc_info:
        await add_item(redis_client, session_id, **_line(MAX_CART_ITEMS))

    assert exc_info.value.session_id == session_id
    assert exc_info.value.limit == MAX_CART_ITEMS
    assert len(await get_cart(redis_client, session_id)) == MAX_CART_ITEMS


@pytest.mark.asyncio
async def test_add_item_rejects_quantity_out_of_range(redis_client: RedisClient) -> None:
    with pytest.raises(ValueError, match="quantity must be between"):
        await add_item(
            redis_client,
            "sess-qty",
            product_id="prod0001",
            name="Product 1",
            price_amount=100.0,
            quantity=0,
        )

    with pytest.raises(ValueError, match="quantity must be between"):
        await add_item(
            redis_client,
            "sess-qty",
            product_id="prod0001",
            name="Product 1",
            price_amount=100.0,
            quantity=100,
        )


@pytest.mark.asyncio
async def test_add_item_rejects_merge_over_max_quantity(redis_client: RedisClient) -> None:
    session_id = "sess-merge-limit"
    await add_item(
        redis_client,
        session_id,
        product_id="prod0001",
        name="Product 1",
        price_amount=100.0,
        quantity=98,
    )

    with pytest.raises(ValueError, match="quantity must be between"):
        await add_item(
            redis_client,
            session_id,
            product_id="prod0001",
            name="Product 1",
            price_amount=100.0,
            quantity=2,
        )


@pytest.mark.asyncio
async def test_remove_item(redis_client: RedisClient) -> None:
    session_id = "sess-remove-1"
    await add_item(
        redis_client,
        session_id,
        product_id="prod0001",
        name="Product 1",
        price_amount=100.0,
    )

    removed = await remove_item(redis_client, session_id, "prod0001")
    assert removed is True
    assert await get_cart(redis_client, session_id) == []

    not_removed = await remove_item(redis_client, session_id, "prod0001")
    assert not_removed is False


@pytest.mark.asyncio
async def test_update_quantity(redis_client: RedisClient) -> None:
    session_id = "sess-update-1"
    await add_item(
        redis_client,
        session_id,
        product_id="prod0001",
        name="Product 1",
        price_amount=100.0,
        quantity=1,
    )

    updated = await update_quantity(redis_client, session_id, "prod0001", 42)
    assert updated.quantity == 42
    assert (await get_cart(redis_client, session_id))[0].quantity == 42


@pytest.mark.asyncio
async def test_update_quantity_missing_item_raises(redis_client: RedisClient) -> None:
    with pytest.raises(CartItemNotFound):
        await update_quantity(redis_client, "sess-missing", "prod9999", 1)


@pytest.mark.asyncio
async def test_update_quantity_rejects_out_of_range(redis_client: RedisClient) -> None:
    session_id = "sess-update-bad"
    await add_item(
        redis_client,
        session_id,
        product_id="prod0001",
        name="Product 1",
        price_amount=100.0,
    )

    with pytest.raises(ValueError, match="quantity must be between"):
        await update_quantity(redis_client, session_id, "prod0001", 0)


@pytest.mark.asyncio
async def test_clear_cart(redis_client: RedisClient) -> None:
    session_id = "sess-clear-1"
    await add_item(
        redis_client,
        session_id,
        product_id="prod0001",
        name="Product 1",
        price_amount=100.0,
    )
    await add_item(
        redis_client,
        session_id,
        product_id="prod0002",
        name="Product 2",
        price_amount=200.0,
    )

    await clear_cart(redis_client, session_id)
    assert await get_cart(redis_client, session_id) == []
