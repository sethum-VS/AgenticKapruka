"""HTMX partial routes for search filters and checkout fragments."""

from __future__ import annotations

import html

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/search", response_class=HTMLResponse)
async def search_by_category(
    category: str = Query(..., min_length=1, max_length=120),
) -> HTMLResponse:
    """Return filtered search results HTML for category chip hx-get swaps into #results."""
    safe_category = html.escape(category.strip())
    body = (
        f'<div data-testid="search-results" data-category="{safe_category}">'
        f'<p class="text-sm text-commerce-muted">Showing {safe_category}</p>'
        "</div>"
    )
    return HTMLResponse(body)


@router.get("")
async def partials_index() -> dict[str, str]:
    """Partials route health stub."""
    return {"status": "ok", "route": "partials"}
