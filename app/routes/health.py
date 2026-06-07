"""Health check routes (stub until PRD-081)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness probe until aggregated health lands in PRD-081."""
    return {"status": "ok"}
