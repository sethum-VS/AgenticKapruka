"""Chat routes (stub until PRD-023+)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def chat_index() -> dict[str, str]:
    """Placeholder chat page endpoint."""
    return {"status": "stub", "route": "chat"}
