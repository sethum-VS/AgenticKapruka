"""Map Kapruka MCP errors and unhandled exceptions to HTMX HTML partials."""

from __future__ import annotations

import logging
from typing import Final

from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from lib.kapruka.errors import (
    KaprukaError,
    KaprukaNotFoundError,
    KaprukaRateLimitError,
    KaprukaValidationError,
)
from lib.redis.rate_limit import RateLimitExceeded, retry_after_header

logger = logging.getLogger(__name__)

_GENERIC_ERROR_TITLE: Final = "Something went wrong"
_GENERIC_ERROR_MESSAGE: Final = "Please try again in a moment."

_FRIENDLY_MESSAGES: Final[dict[str, str]] = {
    "empty_cart": "Your cart is empty. Add items before you checkout.",
    "missing_field": "Please complete all required checkout fields.",
    "past_delivery_date": "Choose a delivery date that is today or later.",
    "product_out_of_stock": "That product is no longer in stock.",
    "city_not_deliverable": "We cannot deliver to that city. Try another location.",
    "date_not_deliverable": "Delivery is not available on that date.",
    "product_not_found": "We could not find that product.",
    "order_not_found": "We could not find an order with that number.",
    "429": "Too many requests. Please wait a moment and try again.",
    "validation_error": "Please check your input and try again.",
}


def human_readable_message(exc: KaprukaError) -> str:
    """Return a user-facing message for a Kapruka MCP error."""
    return _FRIENDLY_MESSAGES.get(exc.code, exc.message)


def _wants_html_response(request: Request) -> bool:
    """True when the client expects an HTMX fragment rather than JSON."""
    if request.headers.get("HX-Request", "").lower() == "true":
        return True
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "application/json" not in accept


def _render_error_banner_html(
    *,
    error_code: str,
    message: str,
    title: str,
) -> str:
    """Lazy import avoids app.templating ↔ graphs circular import at startup."""
    from app.templating import render_error_banner

    return render_error_banner(error_code=error_code, message=message, title=title)


def _html_error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    title: str = "Unable to complete request",
    headers: dict[str, str] | None = None,
) -> HTMLResponse:
    html = _render_error_banner_html(
        error_code=error_code,
        message=message,
        title=title,
    )
    return HTMLResponse(html, status_code=status_code, headers=headers)


def _error_response(
    request: Request,
    *,
    status_code: int,
    error_code: str,
    message: str,
    title: str = "Unable to complete request",
    headers: dict[str, str] | None = None,
) -> Response:
    if _wants_html_response(request):
        return _html_error_response(
            status_code=status_code,
            error_code=error_code,
            message=message,
            title=title,
            headers=headers,
        )
    return JSONResponse(
        status_code=status_code,
        content={"error_code": error_code, "message": message},
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register Kapruka and fallback exception handlers on the FastAPI app."""

    @app.exception_handler(KaprukaValidationError)
    async def handle_kapruka_validation_error(
        request: Request,
        exc: KaprukaValidationError,
    ) -> Response:
        return _error_response(
            request,
            status_code=400,
            error_code=exc.code,
            message=human_readable_message(exc),
        )

    @app.exception_handler(KaprukaNotFoundError)
    async def handle_kapruka_not_found_error(
        request: Request,
        exc: KaprukaNotFoundError,
    ) -> Response:
        return _error_response(
            request,
            status_code=404,
            error_code=exc.code,
            message=human_readable_message(exc),
            title="Not found",
        )

    @app.exception_handler(KaprukaRateLimitError)
    async def handle_kapruka_rate_limit_error(
        request: Request,
        exc: KaprukaRateLimitError,
    ) -> Response:
        return _error_response(
            request,
            status_code=429,
            error_code=exc.code,
            message=human_readable_message(exc),
            title="Rate limit reached",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    @app.exception_handler(RateLimitExceeded)
    async def handle_rate_limit_exceeded(
        request: Request,
        exc: RateLimitExceeded,
    ) -> Response:
        return _error_response(
            request,
            status_code=429,
            error_code="rate_limit_exceeded",
            message="Too many requests. Please wait a moment and try again.",
            title="Rate limit reached",
            headers=retry_after_header(exc),
        )

    @app.exception_handler(KaprukaError)
    async def handle_kapruka_error(request: Request, exc: KaprukaError) -> Response:
        return _error_response(
            request,
            status_code=502,
            error_code=exc.code,
            message=human_readable_message(exc),
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(request: Request, exc: Exception) -> Response:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return _error_response(
            request,
            status_code=500,
            error_code="internal_error",
            message=_GENERIC_ERROR_MESSAGE,
            title=_GENERIC_ERROR_TITLE,
        )
