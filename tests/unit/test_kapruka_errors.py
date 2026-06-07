"""Unit tests for Kapruka MCP error parsing."""

from __future__ import annotations

import pytest

from lib.kapruka.errors import (
    DEFAULT_RETRY_AFTER_SECONDS,
    KaprukaNotFoundError,
    KaprukaRateLimitError,
    KaprukaValidationError,
    parse_mcp_error,
)


def test_parse_mcp_error_429_rate_limit() -> None:
    """429 payloads map to KaprukaRateLimitError with retry_after_seconds."""
    with pytest.raises(KaprukaRateLimitError) as exc_info:
        parse_mcp_error("Error (429): Too many requests. Retry after 30 seconds.")

    exc = exc_info.value
    assert exc.code == "429"
    assert exc.retry_after_seconds == 30
    assert "Too many requests" in exc.message


def test_parse_mcp_error_429_default_retry_when_unparseable() -> None:
    """429 without explicit retry hint falls back to DEFAULT_RETRY_AFTER_SECONDS."""
    with pytest.raises(KaprukaRateLimitError) as exc_info:
        parse_mcp_error("Error (429): Rate limit exceeded.")

    assert exc_info.value.retry_after_seconds == DEFAULT_RETRY_AFTER_SECONDS


def test_parse_mcp_error_empty_cart() -> None:
    """empty_cart maps to KaprukaValidationError."""
    with pytest.raises(KaprukaValidationError) as exc_info:
        parse_mcp_error("Error (empty_cart): Cart must contain at least one item.")

    exc = exc_info.value
    assert exc.code == "empty_cart"
    assert "at least one item" in exc.message


def test_parse_mcp_error_product_not_found() -> None:
    """product_not_found maps to KaprukaNotFoundError (live MCP format)."""
    with pytest.raises(KaprukaNotFoundError) as exc_info:
        parse_mcp_error("Error (product_not_found): No product exists with the given ID")

    exc = exc_info.value
    assert exc.code == "product_not_found"
    assert exc.message == "No product exists with the given ID"


def test_parse_mcp_error_success_json_does_not_raise() -> None:
    """Successful JSON tool payloads pass through without raising."""
    parse_mcp_error('{"results": [{"id": "cake1"}]}')


def test_parse_mcp_error_order_not_found() -> None:
    """order_not_found also maps to KaprukaNotFoundError."""
    with pytest.raises(KaprukaNotFoundError) as exc_info:
        parse_mcp_error("Error (order_not_found): No order exists with the given order number")

    assert exc_info.value.code == "order_not_found"
