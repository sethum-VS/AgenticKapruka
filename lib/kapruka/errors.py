"""Kapruka MCP error parsing and typed exception hierarchy."""

from __future__ import annotations

import re
from typing import Final

_STRUCTURED_ERROR = re.compile(
    r"^Error\s*\((?P<code>[^)]+)\):\s*(?P<message>.+)$",
    re.DOTALL,
)
_SIMPLE_ERROR = re.compile(r"^Error:\s*(?P<message>.+)$", re.DOTALL)
_RETRY_AFTER_SECONDS = re.compile(
    r"retry(?:\s*-\s*after|\s+after)\s+(\d+)\s*seconds?",
    re.IGNORECASE,
)

NOT_FOUND_CODES: Final[frozenset[str]] = frozenset(
    {
        "product_not_found",
        "order_not_found",
    }
)

VALIDATION_CODES: Final[frozenset[str]] = frozenset(
    {
        "empty_cart",
        "missing_field",
        "past_delivery_date",
        "product_out_of_stock",
        "city_not_deliverable",
        "date_not_deliverable",
    }
)

DEFAULT_RETRY_AFTER_SECONDS: Final = 60


class KaprukaError(Exception):
    """Base exception for Kapruka MCP tool error payloads."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"Error ({code}): {message}")


class KaprukaRateLimitError(KaprukaError):
    """Kapruka MCP public-tier rate limit (HTTP 429 equivalent)."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retry_after_seconds: int,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(code, message)


class KaprukaValidationError(KaprukaError):
    """Client input or checkout precondition rejected by Kapruka MCP."""


class KaprukaNotFoundError(KaprukaError):
    """Product, order, or other resource not found."""


def parse_mcp_error(result_str: str) -> None:
    """Raise a typed KaprukaError when *result_str* is an MCP error payload.

    Successful JSON or markdown responses are ignored (no exception raised).
    """
    text = result_str.strip()
    if not text:
        return

    structured = _STRUCTURED_ERROR.match(text)
    if structured is not None:
        code = structured.group("code").strip()
        message = structured.group("message").strip()
        _raise_for_code(code, message)
        return

    simple = _SIMPLE_ERROR.match(text)
    if simple is not None:
        message = simple.group("message").strip()
        raise KaprukaError("unknown", message)

    if text.lower().startswith("error executing tool"):
        raise KaprukaValidationError("validation_error", text)


def _raise_for_code(code: str, message: str) -> None:
    if code == "429":
        raise KaprukaRateLimitError(
            code,
            message,
            retry_after_seconds=_parse_retry_after_seconds(message),
        )

    if code in NOT_FOUND_CODES:
        raise KaprukaNotFoundError(code, message)

    if code in VALIDATION_CODES:
        raise KaprukaValidationError(code, message)

    raise KaprukaError(code, message)


def _parse_retry_after_seconds(message: str) -> int:
    match = _RETRY_AFTER_SECONDS.search(message)
    if match is not None:
        return max(1, int(match.group(1)))
    return DEFAULT_RETRY_AFTER_SECONDS
