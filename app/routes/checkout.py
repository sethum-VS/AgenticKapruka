"""Checkout routes for delivery validation and order flow."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from app.dependencies import get_redis
from app.templating import (
    render_delivery_date_error,
    render_delivery_date_status,
)
from lib.chat.deps import client_ip_from_request, ensure_kapruka_service
from lib.kapruka.errors import KaprukaError
from lib.kapruka.types import CheckDeliveryOutput
from lib.redis.client import RedisClient
from lib.utils.timezone import colombo_today_iso, is_past_colombo_date

router = APIRouter()

RedisDep = Annotated[RedisClient, Depends(get_redis)]

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@router.post("/check-delivery", response_class=HTMLResponse)
async def check_delivery_date(
    request: Request,
    redis_client: RedisDep,
    city: Annotated[str, Form()],
    delivery_date: Annotated[str, Form()],
) -> HTMLResponse:
    """Validate delivery date and city via Kapruka check_delivery; return status partial."""
    city_value = city.strip()
    date_value = delivery_date.strip()

    if not city_value:
        return HTMLResponse(
            render_delivery_date_error(
                title="City required",
                message="Enter a delivery city before choosing a date.",
            )
        )

    if not _ISO_DATE.match(date_value):
        return HTMLResponse(
            render_delivery_date_error(
                title="Invalid date",
                message="Choose a valid delivery date (YYYY-MM-DD).",
            )
        )

    if is_past_colombo_date(date_value):
        return HTMLResponse(
            render_delivery_date_error(
                title="Date in the past",
                message=(
                    f"Please choose today ({colombo_today_iso()}) or a future date. "
                    "Delivery dates use Sri Lanka time (Asia/Colombo)."
                ),
            )
        )

    service = await ensure_kapruka_service(request, redis_client)
    try:
        result: CheckDeliveryOutput = await service.check_delivery(
            client_ip_from_request(request),
            city=city_value,
            delivery_date=date_value,
        )
    except KaprukaError as exc:
        raise HTTPException(status_code=502, detail=exc.message) from exc

    return HTMLResponse(render_delivery_date_status(result=result))


@router.get("")
async def checkout_index() -> dict[str, str]:
    """Placeholder checkout endpoint."""
    return {"status": "stub", "route": "checkout"}
