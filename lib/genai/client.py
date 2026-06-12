"""Factory for google-genai clients (Vertex AI ADC or Gemini API key)."""

from __future__ import annotations

from typing import Literal

from google import genai
from google.genai import types

from app.config import Settings, get_settings

GeminiBackend = Literal["vertex", "api_key"]

# Google Gen AI SDK defaults to 3 retries (~1s, 2s, 4s). Vertex 429s are often
# transient shared-capacity saturation; use longer exponential backoff per:
# https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429
VERTEX_HTTP_OPTIONS = types.HttpOptions(
    retry_options=types.HttpRetryOptions(
        attempts=8,
        initial_delay=2.0,
        max_delay=60.0,
        exp_base=2.0,
        jitter=1.0,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    ),
)


def create_genai_client(
    *,
    settings: Settings | None = None,
    api_key: str | None = None,
    backend: GeminiBackend | None = None,
    location: str | None = None,
) -> genai.Client:
    """Build a google-genai client.

    Default backend is Vertex AI using Application Default Credentials
    (``gcloud auth application-default login`` or ``GOOGLE_APPLICATION_CREDENTIALS``).
    Set ``GEMINI_BACKEND=api_key`` and ``GOOGLE_API_KEY`` to use the Gemini Developer API.
    """
    if api_key is not None:
        return genai.Client(api_key=api_key)

    cfg = settings or get_settings()
    resolved_backend = backend or cfg.gemini_backend

    if resolved_backend == "api_key":
        if not cfg.google_api_key:
            msg = "GOOGLE_API_KEY is required when GEMINI_BACKEND=api_key"
            raise ValueError(msg)
        return genai.Client(api_key=cfg.google_api_key)

    resolved_location = location or cfg.gemini_chat_location
    return genai.Client(
        vertexai=True,
        project=cfg.gcp_project_id,
        location=resolved_location,
        http_options=VERTEX_HTTP_OPTIONS,
    )
