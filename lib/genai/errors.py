"""Shared helpers for NVIDIA NIM / OpenAI SDK error handling."""

from __future__ import annotations

from openai import RateLimitError


def is_rate_limited(exc: BaseException) -> bool:
    """Return True for NVIDIA NIM 429 rate limit errors."""
    if isinstance(exc, RateLimitError):
        return True
    return False
