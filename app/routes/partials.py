"""HTMX partial routes (stub until PRD-052+)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def partials_index() -> dict[str, str]:
    """Placeholder partials endpoint."""
    return {"status": "stub", "route": "partials"}
