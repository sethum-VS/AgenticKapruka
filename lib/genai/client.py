"""Factory for google-genai clients (Vertex AI ADC or Gemini API key)."""

from __future__ import annotations

from typing import Literal

from google import genai

from app.config import Settings, get_settings

GeminiBackend = Literal["vertex", "api_key"]


def create_genai_client(
    *,
    settings: Settings | None = None,
    api_key: str | None = None,
    backend: GeminiBackend | None = None,
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

    return genai.Client(
        vertexai=True,
        project=cfg.gcp_project_id,
        location=cfg.gcp_location,
    )
