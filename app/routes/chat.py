"""Chat page routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import Response

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
