"""Chat page routes."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from langchain_core.runnables import RunnableConfig
from starlette.responses import Response, StreamingResponse

from app.dependencies import get_redis
from app.templating import get_templates, render_cart_partial_oob
from lib.chat.deps import (
    build_shopping_graph_deps,
    client_ip_from_request,
    get_compiled_chat_graph,
    resolve_turn_state,
)
from lib.chat.page_context import (
    cart_template_context,
    currency_template_context,
    resolve_page_cart,
    resolve_page_currency,
)
from lib.chat.session import (
    SESSION_COOKIE_NAME,
    cookie_params,
    resolve_chat_thread_id,
    rotate_chat_thread,
)
from lib.chat.sse import format_sse_event
from lib.chat.streaming import iter_chat_sse_events
from lib.debug.trace import is_debug_trace_enabled, trace_error, trace_turn_start
from lib.redis.cart import clear_cart
from lib.redis.client import RedisClient
from lib.redis.session import get_session_currency
from lib.zep.session import get_or_create_session

logger = logging.getLogger(__name__)

router = APIRouter()

RedisDep = Annotated[RedisClient, Depends(get_redis)]

def _chat_new_empty_state_html() -> str:
    """OOB swap HTML that restores the welcome empty state after new session."""
    templates = get_templates()
    empty = templates.get_template("chat/empty_state.html").render()
    return f'<div id="chat-messages" hx-swap-oob="innerHTML">{empty}</div>'


_STREAM_SETUP_ERROR_HTML = (
    '<div class="mb-4 flex flex-col gap-4" data-role="assistant-message">'
    '<div class="flex items-start gap-3">'
    '<div class="max-w-[85%] rounded-xl rounded-tl-none border border-error-container '
    'bg-error-container/30 p-4 text-body-md text-on-error-container" role="alert">'
    "Something went wrong. Please try again.</div></div></div>"
)


def _render_user_turn_html(message: str) -> str:
    """User bubble plus OOB removal of the welcome empty state."""
    templates = get_templates()
    user_bubble = templates.get_template("chat/message_user.html").render(message=message)
    oob_remove = '<div id="chat-empty-state" hx-swap-oob="delete"></div>'
    return oob_remove + user_bubble


async def _chat_event_stream(
    *,
    request: Request,
    redis_client: RedisClient,
    message: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """Run the shopping graph and yield SSE HTML events."""
    deps = await build_shopping_graph_deps(request, redis_client)
    graph = await get_compiled_chat_graph(redis_client, deps=deps)

    zep_thread_id: str | None = thread_id
    zep_client = deps.zep_client
    if zep_client is not None:
        zep_thread_id = await get_or_create_session(redis_client, zep_client, thread_id)

    config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    currency = await get_session_currency(redis_client, thread_id)
    state = await resolve_turn_state(
        graph,
        message=message,
        session_id=thread_id,
        zep_thread_id=zep_thread_id,
        config=config,
        currency=currency,
    )
    trace_turn_start(
        thread_id=thread_id,
        message=message,
        currency=currency,
        client_ip=client_ip_from_request(request),
        state=dict(state),
    )
    user_html = _render_user_turn_html(message)

    async for event in iter_chat_sse_events(
        graph=graph,
        state=state,
        config=config,
        user_html=user_html,
    ):
        yield event


@router.get("")
async def chat_index(request: Request, redis_client: RedisDep) -> Response:
    """Full-screen chat viewport with welcome empty state."""
    templates = get_templates()
    currency = await resolve_page_currency(request, redis_client)
    cart_items = await resolve_page_cart(request, redis_client)
    return templates.TemplateResponse(
        request,
        "chat/index.html",
        {
            "title": "Chat — AgenticKapruka",
            "debug_trace": is_debug_trace_enabled(),
            **currency_template_context(currency),
            **cart_template_context(cart_items),
        },
    )


@router.post("/stream")
async def chat_stream(
    request: Request,
    redis_client: RedisDep,
    message: str = Form(..., min_length=1, max_length=2000),
) -> Response:
    """Stream LangGraph assistant HTML fragments as Server-Sent Events."""
    stripped = message.strip()
    if not stripped:
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    thread_id, new_cookie = resolve_chat_thread_id(request)

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for payload in _chat_event_stream(
                request=request,
                redis_client=redis_client,
                message=stripped,
                thread_id=thread_id,
            ):
                yield payload
        except Exception as exc:
            trace_error(f"chat stream setup failed (session={thread_id})", exc)
            logger.exception("chat stream failed for session %s", thread_id)
            yield format_sse_event(_STREAM_SETUP_ERROR_HTML)
            yield format_sse_event("", event="done")

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    response = StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )
    if new_cookie is not None:
        response.set_cookie(SESSION_COOKIE_NAME, new_cookie, **cookie_params())
    return response


@router.post("/new")
async def chat_new(request: Request, redis_client: RedisDep) -> Response:
    """Rotate chat thread and clear conversation context; cart is cleared."""
    prior_thread_id, new_thread_id, signed_cookie = rotate_chat_thread(request)
    await clear_cart(redis_client, new_thread_id)
    if prior_thread_id and prior_thread_id != new_thread_id:
        await clear_cart(redis_client, prior_thread_id)
    currency = await get_session_currency(redis_client, new_thread_id)
    content = _chat_new_empty_state_html() + render_cart_partial_oob(
        items=[],
        currency=currency,
    )
    response = Response(content=content, media_type="text/html")
    response.set_cookie(SESSION_COOKIE_NAME, signed_cookie, **cookie_params())
    return response
