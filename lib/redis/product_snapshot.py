"""Short-TTL product snapshots from search results for cart-add fallback."""

from __future__ import annotations

import json
import logging
from typing import Final

from lib.kapruka.types import (
    GetProductOutput,
    ProductAttributes,
    ProductResult,
    ProductShipping,
)
from lib.redis.client import RedisClient

logger = logging.getLogger(__name__)

PRODUCT_SNAPSHOT_TTL: Final = 1800
_KEY_PREFIX: Final = "product_snapshot"


def _snapshot_key(product_id: str, currency: str) -> str:
    return f"{_KEY_PREFIX}:{product_id}:{currency.upper()}"


def product_result_to_detail(product: ProductResult) -> GetProductOutput:
    """Lift a search hit into a GetProductOutput shape for cart add."""
    images = [product.image_url] if product.image_url else []
    summary = product.summary or product.name
    return GetProductOutput(
        id=product.id,
        name=product.name,
        description=summary,
        summary=summary,
        price=product.price,
        compare_at_price=product.compare_at_price,
        in_stock=product.in_stock,
        stock_level=product.stock_level,
        category=product.category,
        variants=[],
        images=images,
        attributes=ProductAttributes(),
        shipping=ProductShipping(
            ships_from="Colombo",
            ships_internationally=product.ships_internationally,
            restricted_countries=[],
        ),
        rating=product.rating,
        url=product.url,
    )


async def cache_search_product_snapshots(
    redis_client: RedisClient,
    products: list[ProductResult],
    *,
    currency: str,
    ttl: int = PRODUCT_SNAPSHOT_TTL,
) -> None:
    """Store search hits so cart add can succeed if live get_product flaps."""
    if not products:
        return
    currency_upper = currency.upper()
    pipe = redis_client.client.pipeline()
    for product in products:
        if not product.id:
            continue
        detail = product_result_to_detail(product)
        key = _snapshot_key(product.id, currency_upper)
        pipe.set(key, detail.model_dump_json(), ex=ttl)
    try:
        await pipe.execute()
    except Exception:
        logger.debug("product snapshot cache write failed", exc_info=True)


async def get_product_snapshot(
    redis_client: RedisClient,
    product_id: str,
    *,
    currency: str,
) -> GetProductOutput | None:
    """Return a cached search snapshot for product_id, if present."""
    key = _snapshot_key(product_id, currency)
    try:
        raw = await redis_client.client.get(key)
    except Exception:
        logger.debug("product snapshot cache read failed", exc_info=True)
        return None
    if not raw:
        return None
    try:
        return GetProductOutput.model_validate(json.loads(raw))
    except Exception:
        logger.debug("product snapshot cache decode failed for %s", product_id, exc_info=True)
        return None
