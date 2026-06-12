"""Shared helpers for google-genai / Vertex error handling."""

from __future__ import annotations

from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors


def is_resource_exhausted(exc: BaseException) -> bool:
    """Return True for Vertex/Gemini 429 RESOURCE_EXHAUSTED errors."""
    if isinstance(exc, google_exceptions.ResourceExhausted):
        return True
    if isinstance(exc, genai_errors.ClientError):
        if exc.code == 429:
            return True
        if exc.status == "RESOURCE_EXHAUSTED":
            return True
    return False
