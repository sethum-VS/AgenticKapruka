"""Chat page routes."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from starlette.responses import HTMLResponse, Response

from app.templating import get_templates

router = APIRouter()


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
    message: str = Form(..., min_length=1, max_length=2000),
) -> Response:
    """Accept chat message and return user bubble HTML for HTMX swap.

    Stub until PRD-037 wires LangGraph SSE streaming.
    """
    stripped = message.strip()
    if not stripped:
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    templates = get_templates()
    user_bubble = templates.get_template("chat/message_user.html").render(
        message=stripped,
    )
    # Remove welcome empty state on first message via OOB swap.
    oob_remove = '<div id="chat-empty-state" hx-swap-oob="delete"></div>'
    return HTMLResponse(content=oob_remove + user_bubble)
