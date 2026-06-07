"""Checkout routes (stub until PRD-060+)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def checkout_index() -> dict[str, str]:
    """Placeholder checkout endpoint."""
    return {"status": "stub", "route": "checkout"}
