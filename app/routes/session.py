"""Session preference routes (currency, etc.)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from app.dependencies import get_redis
from app.templating import render_cart_partial_oob
from lib.cart.pricing import refresh_cart_prices_for_currency
from lib.chat.deps import client_ip_from_request, ensure_kapruka_service
from lib.chat.session import SESSION_COOKIE_NAME, cookie_params, resolve_chat_thread_id
from lib.redis.cart import get_cart
from lib.redis.client import RedisClient
from lib.redis.session import set_session_currency

router = APIRouter()

RedisDep = Annotated[RedisClient, Depends(get_redis)]


@router.post("/currency")
async def update_session_currency(
    request: Request,
    redis_client: RedisDep,
    currency: str = Form(...),
) -> Response:
    """Persist currency preference; refresh cart line prices when the cart is non-empty."""
    thread_id, new_cookie = resolve_chat_thread_id(request)
    try:
        stored_currency = await set_session_currency(redis_client, thread_id, currency)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    items = await get_cart(redis_client, thread_id)
    if items:
        kapruka_service = await ensure_kapruka_service(request, redis_client)
        items = await refresh_cart_prices_for_currency(
            redis_client,
            thread_id,
            currency=stored_currency,
            kapruka_service=kapruka_service,
            client_ip=client_ip_from_request(request),
        )
        response: Response = HTMLResponse(
            render_cart_partial_oob(items=items, currency=stored_currency),
        )
    else:
        response = Response(status_code=204)

    if new_cookie is not None:
        response.set_cookie(SESSION_COOKIE_NAME, new_cookie, **cookie_params())
    return response
