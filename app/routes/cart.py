"""Cart HTMX routes — partial swaps into #cart-panel without full page reload."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from app.dependencies import get_redis
from app.templating import render_cart_partial
from lib.chat.deps import client_ip_from_request, ensure_kapruka_service
from lib.chat.session import SESSION_COOKIE_NAME, cookie_params, resolve_chat_thread_id
from lib.kapruka.errors import KaprukaError, KaprukaNotFoundError
from lib.redis.cart import (
    CartItemNotFound,
    CartLimitExceeded,
    add_item,
    get_cart,
    remove_item,
    update_quantity,
)
from lib.redis.client import RedisClient
from lib.redis.session import get_session_currency

router = APIRouter()

RedisDep = Annotated[RedisClient, Depends(get_redis)]


def _cart_html_response(*, html: str, new_cookie: str | None) -> HTMLResponse:
    response = HTMLResponse(html)
    if new_cookie is not None:
        response.set_cookie(SESSION_COOKIE_NAME, new_cookie, **cookie_params())
    return response


async def _cart_partial_response(
    redis_client: RedisClient,
    session_id: str,
    *,
    new_cookie: str | None,
) -> HTMLResponse:
    currency = await get_session_currency(redis_client, session_id)
    items = await get_cart(redis_client, session_id)
    html = render_cart_partial(items=items, currency=currency)
    return _cart_html_response(html=html, new_cookie=new_cookie)


@router.get("/panel", response_class=HTMLResponse)
async def cart_panel(
    request: Request,
    redis_client: RedisDep,
) -> HTMLResponse:
    """Return the cart partial — used to refresh the drawer when it opens."""
    thread_id, new_cookie = resolve_chat_thread_id(request)
    return await _cart_partial_response(redis_client, thread_id, new_cookie=new_cookie)


@router.post("/add", response_class=HTMLResponse)
async def cart_add(
    request: Request,
    redis_client: RedisDep,
    product_id: str = Form(...),
    quantity: int = Form(1),
    icing_text: str | None = Form(None),
) -> HTMLResponse:
    """Add or merge a product line; return refreshed cart partial for outerHTML swap."""
    thread_id, new_cookie = resolve_chat_thread_id(request)
    currency = await get_session_currency(redis_client, thread_id)
    service = await ensure_kapruka_service(request, redis_client)

    try:
        product = await service.get_product(
            client_ip_from_request(request),
            product_id=product_id,
            currency=currency,
        )
    except KaprukaNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc
    except KaprukaError as exc:
        raise HTTPException(status_code=502, detail=exc.message) from exc

    if not product.in_stock:
        raise HTTPException(status_code=422, detail="Product is out of stock")

    price_amount = product.price.amount
    if price_amount is None:
        raise HTTPException(status_code=422, detail="Product price is unavailable")

    try:
        await add_item(
            redis_client,
            thread_id,
            product_id=product_id,
            name=product.name,
            price_amount=price_amount,
            price_currency=product.price.currency,
            quantity=quantity,
            icing_text=icing_text,
        )
    except CartLimitExceeded as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await _cart_partial_response(redis_client, thread_id, new_cookie=new_cookie)


@router.post("/remove", response_class=HTMLResponse)
async def cart_remove(
    request: Request,
    redis_client: RedisDep,
    product_id: str = Form(...),
) -> HTMLResponse:
    """Remove a line item and return the updated cart partial."""
    thread_id, new_cookie = resolve_chat_thread_id(request)
    await remove_item(redis_client, thread_id, product_id)
    return await _cart_partial_response(redis_client, thread_id, new_cookie=new_cookie)


@router.post("/update", response_class=HTMLResponse)
async def cart_update(
    request: Request,
    redis_client: RedisDep,
    product_id: str = Form(...),
    quantity: int = Form(...),
) -> HTMLResponse:
    """Set quantity (or remove when below 1) and return the updated cart partial."""
    thread_id, new_cookie = resolve_chat_thread_id(request)

    try:
        if quantity < 1:
            await remove_item(redis_client, thread_id, product_id)
        else:
            await update_quantity(redis_client, thread_id, product_id, quantity)
    except CartItemNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await _cart_partial_response(redis_client, thread_id, new_cookie=new_cookie)
