"""Cart price refresh when the shopper changes display currency."""

from __future__ import annotations

import logging

from lib.kapruka.errors import KaprukaError
from lib.kapruka.service import KaprukaService
from lib.redis.cart import StoredCartItem, cart_key, get_cart
from lib.redis.client import RedisClient
from lib.redis.session import SESSION_TTL_SECONDS

logger = logging.getLogger(__name__)


async def refresh_cart_prices_for_currency(
    redis_client: RedisClient,
    session_id: str,
    *,
    currency: str,
    kapruka_service: KaprukaService,
    client_ip: str,
) -> list[StoredCartItem]:
    """Re-fetch live Kapruka prices for each cart line in ``currency`` and persist snapshots."""
    items = await get_cart(redis_client, session_id)
    if not items:
        return items

    key = cart_key(session_id)
    refreshed: list[StoredCartItem] = []
    updated_any = False

    for item in items:
        if item.price_currency == currency:
            refreshed.append(item)
            continue
        try:
            product = await kapruka_service.get_product(
                client_ip,
                product_id=item.product_id,
                currency=currency,
            )
        except KaprukaError:
            logger.warning(
                "refresh_cart_prices_for_currency: Kapruka lookup failed for %s",
                item.product_id,
                exc_info=True,
            )
            refreshed.append(item)
            continue

        amount = product.price.amount
        if amount is None:
            refreshed.append(item)
            continue

        updated = item.model_copy(
            update={
                "name": product.name,
                "price_amount": float(amount),
                "price_currency": product.price.currency,
            },
        )
        await redis_client.client.hset(key, item.product_id, updated.model_dump_json())  # type: ignore[misc]
        refreshed.append(updated)
        updated_any = True

    if updated_any:
        await redis_client.client.expire(key, SESSION_TTL_SECONDS)

    return refreshed
