"""Session preference routes (currency, etc.)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from starlette.responses import Response

from app.dependencies import get_redis
from lib.chat.session import SESSION_COOKIE_NAME, cookie_params, resolve_chat_thread_id
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
    """Persist the shopper's currency choice for this browser session."""
    thread_id, new_cookie = resolve_chat_thread_id(request)
    try:
        await set_session_currency(redis_client, thread_id, currency)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = Response(status_code=204)
    if new_cookie is not None:
        response.set_cookie(SESSION_COOKIE_NAME, new_cookie, **cookie_params())
    return response
