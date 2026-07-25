"""Shared helpers for NVIDIA NIM / OpenAI SDK error handling."""

from __future__ import annotations

from openai import APIConnectionError, APITimeoutError, RateLimitError


def is_rate_limited(exc: BaseException) -> bool:
    """Return True for NVIDIA NIM 429 rate limit errors."""
    return isinstance(exc, RateLimitError)


def is_transient_nim_error(exc: BaseException) -> bool:
    """Return True for NIM errors that should degrade gracefully (timeout/conn/429)."""
    return isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError))
