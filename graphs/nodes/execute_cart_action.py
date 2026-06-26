"""Execute add-to-cart against Redis after product resolution."""

from __future__ import annotations

import logging
from typing import Any

from graphs.state import AgentState
from lib.kapruka.errors import KaprukaNotFoundError, KaprukaValidationError
from lib.kapruka.service import KaprukaService
from lib.redis.cart import CartLimitExceeded, add_item, get_cart
from lib.redis.client import RedisClient

logger = logging.getLogger(__name__)


def _price_from_product(product: dict[str, Any]) -> tuple[float | None, str]:
    raw_price = product.get("price")
    price: dict[str, Any] = raw_price if isinstance(raw_price, dict) else {}
    amount = price.get("amount")
    currency = str(price.get("currency") or "LKR")
    if amount is None:
        return None, currency
    return float(amount), currency


async def execute_cart_action(
    state: AgentState,
    *,
    redis_client: RedisClient | None = None,
    kapruka_service: KaprukaService | None = None,
    client_ip: str | None = None,
) -> dict[str, Any]:
    """LangGraph node: add resolved product to the session Redis cart."""
    action = dict(state.get("cart_action_result") or {})
    status = action.get("status")

    if status == "clarify":
        return {}

    if status != "resolved":
        return {
            "cart_action_result": {
                **action,
                "status": "error",
                "message": "I could not determine which product to add.",
            },
        }

    product = action.get("product")
    if not isinstance(product, dict):
        return {
            "cart_action_result": {
                **action,
                "status": "error",
                "message": "I could not determine which product to add.",
            },
        }

    product_id = str(product.get("id") or "").strip()
    if not product_id:
        return {
            "cart_action_result": {
                **action,
                "status": "error",
                "message": "That product is missing an identifier.",
            },
        }

    session_id = state.get("session_id") or ""
    if not redis_client or not session_id:
        return {
            "cart_action_result": {
                **action,
                "status": "error",
                "message": "Your session expired. Refresh the page and try again.",
            },
        }

    currency = state.get("currency") or "LKR"
    name = str(product.get("name") or "Gift")
    snapshot_in_stock = product.get("in_stock")
    in_stock = snapshot_in_stock
    price_amount, price_currency = _price_from_product(product)
    stock_mismatch = False

    if kapruka_service is not None:
        try:
            live = await kapruka_service.get_product(
                client_ip or "127.0.0.1",
                product_id=product_id,
                currency=currency,
            )
            name = live.name
            in_stock = live.in_stock
            if live.price.amount is not None:
                price_amount = float(live.price.amount)
                price_currency = live.price.currency
        except KaprukaNotFoundError:
            return {
                "cart_action_result": {
                    **action,
                    "status": "error",
                    "message": "That product is no longer available on Kapruka.",
                },
            }
        except KaprukaValidationError as exc:
            if exc.code == "product_out_of_stock" and snapshot_in_stock is True:
                stock_mismatch = True
                in_stock = True
                logger.warning(
                    "execute_cart_action: live stock mismatch for %s — snapshot in_stock=True",
                    product_id,
                )
            else:
                return {
                    "cart_action_result": {
                        **action,
                        "status": "error",
                        "message": exc.message,
                        "error_code": exc.code,
                    },
                }

    if in_stock is False and not stock_mismatch:
        return {
            "cart_action_result": {
                **action,
                "status": "error",
                "message": "That product is out of stock.",
                "error_code": "product_out_of_stock",
            },
        }

    if price_amount is None:
        return {
            "cart_action_result": {
                **action,
                "status": "error",
                "message": "That product price is unavailable right now.",
            },
        }

    prior_cart = await get_cart(redis_client, session_id)
    prior_qty = next(
        (row.quantity for row in prior_cart if row.product_id == product_id),
        0,
    )

    try:
        stored = await add_item(
            redis_client,
            session_id,
            product_id=product_id,
            name=name,
            price_amount=price_amount,
            price_currency=price_currency,
            quantity=1,
        )
    except CartLimitExceeded:
        return {
            "cart_action_result": {
                **action,
                "status": "error",
                "message": "Your cart is full — remove an item before adding another.",
                "error_code": "cart_limit_exceeded",
            },
        }
    except ValueError as exc:
        return {
            "cart_action_result": {
                **action,
                "status": "error",
                "message": str(exc),
            },
        }

    cart_rows = await get_cart(redis_client, session_id)
    merged = prior_qty > 0
    logger.info(
        "execute_cart_action: session=%s product=%s qty=%d merged=%s",
        session_id,
        product_id,
        stored.quantity,
        merged,
    )

    result_payload: dict[str, Any] = {
        **action,
        "status": "added",
        "product_name": name,
        "quantity": stored.quantity,
        "merged": merged,
        "cart_items": [row.model_dump() for row in cart_rows],
    }
    if stock_mismatch:
        result_payload["stock_warning"] = (
            "Live stock check disagreed with search results — added from catalog snapshot. "
            "If checkout fails, refresh and try again."
        )

    return {"cart_action_result": result_payload}
