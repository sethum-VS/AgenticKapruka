"""Cart routes (stub until PRD-059+)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def cart_index() -> dict[str, str]:
    """Placeholder cart endpoint."""
    return {"status": "stub", "route": "cart"}
