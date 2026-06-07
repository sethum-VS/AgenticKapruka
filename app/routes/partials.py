"""HTMX partial routes for search filters and checkout fragments."""

from __future__ import annotations

import html
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from app.dependencies import get_redis
from app.templating import render_delivery_city_suggestions
from lib.chat.deps import client_ip_from_request, ensure_kapruka_service
from lib.kapruka.errors import KaprukaError
from lib.redis.client import RedisClient

router = APIRouter()

RedisDep = Annotated[RedisClient, Depends(get_redis)]


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


@router.get("/delivery-cities", response_class=HTMLResponse)
async def delivery_city_suggestions(
    request: Request,
    redis_client: RedisDep,
    q: str = Query("", max_length=50),
) -> HTMLResponse:
    """Return city suggestion li items for delivery city autocomplete hx-get swaps."""
    query = q.strip()
    if len(query) < 2:
        return HTMLResponse("")

    service = await ensure_kapruka_service(request, redis_client)
    try:
        cities = await service.list_delivery_cities(
            client_ip_from_request(request),
            query=query,
            limit=10,
        )
    except KaprukaError as exc:
        raise HTTPException(status_code=502, detail=exc.message) from exc

    return HTMLResponse(render_delivery_city_suggestions(cities=cities))


@router.get("")
async def partials_index() -> dict[str, str]:
    """Partials route health stub."""
    return {"status": "ok", "route": "partials"}
