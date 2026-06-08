"""Tests for Kapruka MCP exception middleware."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.templating import _create_templates, render_error_banner
from lib.kapruka.errors import (
    KaprukaError,
    KaprukaNotFoundError,
    KaprukaRateLimitError,
    KaprukaValidationError,
)
from lib.redis.rate_limit import RateLimitExceeded


@pytest.fixture(autouse=True)
def clear_templates_cache() -> None:
    _create_templates.cache_clear()
    yield
    _create_templates.cache_clear()


def test_error_banner_renders_code_and_message() -> None:
    """Error partial exposes error code and human-readable message."""
    html = render_error_banner(
        error_code="empty_cart",
        message="Your cart is empty. Add items before you checkout.",
        title="Unable to complete request",
    )
    assert 'data-testid="error-banner"' in html
    assert 'data-error-code="empty_cart"' in html
    assert "Your cart is empty" in html
    assert 'role="alert"' in html


@pytest.mark.asyncio
async def test_kapruka_validation_error_returns_400_html_not_json() -> None:
    """KaprukaValidationError yields 400 HTML for HTMX clients, not JSON."""
    application = create_app()

    @application.get("/_test/validation-error", include_in_schema=False)
    async def trigger_validation_error() -> None:
        raise KaprukaValidationError("empty_cart", "Cart cannot be empty")

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/_test/validation-error",
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 400
    assert "text/html" in response.headers["content-type"]
    assert "application/json" not in response.headers["content-type"]
    assert 'data-error-code="empty_cart"' in response.text
    assert "Your cart is empty" in response.text


@pytest.mark.asyncio
async def test_kapruka_validation_error_returns_json_for_api_clients() -> None:
    """Non-HTMX clients still receive structured JSON error payloads."""
    application = create_app()

    @application.get("/_test/validation-error-json", include_in_schema=False)
    async def trigger_validation_error() -> None:
        raise KaprukaValidationError("missing_field", "Recipient phone is required")

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/_test/validation-error-json",
            headers={"Accept": "application/json"},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "missing_field"
    assert "required checkout fields" in payload["message"]


@pytest.mark.asyncio
async def test_kapruka_rate_limit_error_returns_429_with_retry_after() -> None:
    """KaprukaRateLimitError maps to 429 HTML with Retry-After header."""
    application = create_app()

    @application.get("/_test/rate-limit", include_in_schema=False)
    async def trigger_rate_limit() -> None:
        raise KaprukaRateLimitError(
            "429",
            "Rate limit exceeded; retry after 45 seconds",
            retry_after_seconds=45,
        )

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/_test/rate-limit",
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "45"
    assert 'data-error-code="429"' in response.text


@pytest.mark.asyncio
async def test_kapruka_not_found_error_returns_404_html() -> None:
    """KaprukaNotFoundError maps to 404 HTML partial."""
    application = create_app()

    @application.get("/_test/not-found", include_in_schema=False)
    async def trigger_not_found() -> None:
        raise KaprukaNotFoundError("product_not_found", "No such product")

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/_test/not-found",
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 404
    assert 'data-error-code="product_not_found"' in response.text


@pytest.mark.asyncio
async def test_rate_limit_exceeded_returns_429_html() -> None:
    """App-level RateLimitExceeded maps to 429 HTML with Retry-After."""
    application = create_app()

    @application.get("/_test/app-rate-limit", include_in_schema=False)
    async def trigger_app_rate_limit() -> None:
        raise RateLimitExceeded(
            30,
            limit_type="global",
            ip="127.0.0.1",
            tool_name="kapruka_search_products",
        )

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/_test/app-rate-limit",
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    assert 'data-error-code="rate_limit_exceeded"' in response.text


@pytest.mark.asyncio
async def test_unhandled_exception_returns_500_html_without_details() -> None:
    """Unhandled exceptions return a generic HTML banner without leaking internals."""
    application = create_app()

    @application.get("/_test/boom", include_in_schema=False)
    async def trigger_boom() -> None:
        raise RuntimeError("database connection lost")

    # ServerErrorMiddleware always re-raises after handling Exception subclasses.
    transport = ASGITransport(app=application, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/_test/boom",
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 500
    assert 'data-error-code="internal_error"' in response.text
    assert "database connection lost" not in response.text
    assert "Please try again" in response.text


@pytest.mark.asyncio
async def test_generic_kapruka_error_returns_502_html() -> None:
    """Unhandled KaprukaError subclasses map to 502 HTML."""
    application = create_app()

    @application.get("/_test/kapruka-error", include_in_schema=False)
    async def trigger_kapruka_error() -> None:
        raise KaprukaError("upstream_error", "Kapruka service unavailable")

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/_test/kapruka-error",
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 502
    assert 'data-error-code="upstream_error"' in response.text
