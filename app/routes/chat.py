"""Chat page routes."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from langchain_core.runnables import RunnableConfig
from starlette.responses import Response, StreamingResponse

from app.dependencies import get_redis
from app.templating import get_templates
from lib.chat.deps import (
    build_shopping_graph_deps,
    get_compiled_chat_graph,
    resolve_turn_state,
)
from lib.chat.session import SESSION_COOKIE_NAME, cookie_params, resolve_chat_thread_id
from lib.chat.sse import format_sse_event
from lib.chat.streaming import iter_chat_sse_events
from lib.redis.client import RedisClient
from lib.zep.session import get_or_create_session

logger = logging.getLogger(__name__)

router = APIRouter()

RedisDep = Annotated[RedisClient, Depends(get_redis)]

_STREAM_SETUP_ERROR_HTML = (
    '<div class="flex justify-start">'
    '<div class="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 '
    'text-sm text-red-800" role="alert">'
    "Something went wrong. Please try again.</div></div>"
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
    state = await resolve_turn_state(
        graph,
        message=message,
        session_id=thread_id,
        zep_thread_id=zep_thread_id,
        config=config,
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
async def chat_index(request: Request) -> Response:
    """Full-screen chat viewport with welcome empty state."""
    templates = get_templates()
    return templates.TemplateResponse(
        request,
        "chat/index.html",
        {"title": "Chat — AgenticKapruka"},
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
        except Exception:
            logger.exception("chat stream failed for session %s", thread_id)
            yield format_sse_event(_STREAM_SETUP_ERROR_HTML)

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
