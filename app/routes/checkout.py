"""Checkout routes for delivery validation and order flow."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse

from app.dependencies import get_redis
from app.templating import (
    render_delivery_date_error,
    render_delivery_date_status,
    render_delivery_form_validation_response,
    render_recipient_form_validation_response,
    render_sender_form_validation_response,
)
from lib.chat.deps import client_ip_from_request, ensure_kapruka_service
from lib.checkout.delivery import DeliveryFormValues, parse_delivery_form
from lib.checkout.recipient import RecipientFormValues, parse_recipient_form
from lib.checkout.sender import SenderFormValues, parse_sender_form
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
    result: CheckDeliveryOutput = await service.check_delivery(
        client_ip_from_request(request),
        city=city_value,
        delivery_date=date_value,
    )

    return HTMLResponse(render_delivery_date_status(result=result))


@router.post("/validate-delivery", response_class=HTMLResponse)
async def validate_delivery_form(
    address: Annotated[str, Form()] = "",
    city: Annotated[str, Form()] = "",
    location_type: Annotated[str, Form()] = "house",
    date: Annotated[str, Form()] = "",
    instructions: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Validate delivery address form fields; return form with OOB field errors on failure."""
    values = DeliveryFormValues(
        address=address,
        city=city,
        location_type=location_type,
        date=date,
        instructions=instructions,
    )
    _delivery, errors = parse_delivery_form(values)
    return HTMLResponse(
        render_delivery_form_validation_response(
            values=values,
            errors=errors,
            valid=not errors,
        )
    )


@router.post("/validate-recipient", response_class=HTMLResponse)
async def validate_recipient_form(
    name: Annotated[str, Form()] = "",
    phone: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Validate recipient name and phone; return form with OOB field errors on failure."""
    values = RecipientFormValues(name=name, phone=phone)
    _recipient, errors = parse_recipient_form(values)
    return HTMLResponse(
        render_recipient_form_validation_response(
            values=values,
            errors=errors,
            valid=not errors,
        )
    )


@router.post("/validate-sender", response_class=HTMLResponse)
async def validate_sender_form(
    name: Annotated[str, Form()] = "",
    anonymous: Annotated[str | None, Form()] = None,
) -> HTMLResponse:
    """Validate sender name and anonymous flag; return form with OOB field errors on failure."""
    values = SenderFormValues(
        name=name,
        anonymous=anonymous in ("true", "on", "1"),
    )
    _sender, errors = parse_sender_form(values)
    return HTMLResponse(
        render_sender_form_validation_response(
            values=values,
            errors=errors,
            valid=not errors,
        )
    )


@router.get("")
async def checkout_index() -> dict[str, str]:
    """Placeholder checkout endpoint."""
    return {"status": "stub", "route": "checkout"}
