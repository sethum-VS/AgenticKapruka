"""Factory for NVIDIA NIM OpenAI-compatible client."""

from __future__ import annotations

from openai import OpenAI

from app.config import Settings, get_settings

_client: OpenAI | None = None


def create_nvidia_client(*, settings: Settings | None = None) -> OpenAI:
    """Return a cached NVIDIA NIM OpenAI client singleton.

    Points at ``NVIDIA_BASE_URL`` (default https://integrate.api.nvidia.com/v1)
    with ``NVIDIA_API_KEY`` for authentication.
    """
    global _client
    if _client is not None:
        return _client
    cfg = settings or get_settings()
    _client = OpenAI(
        base_url=cfg.nvidia_base_url,
        api_key=cfg.nvidia_api_key,
        timeout=30.0,
    )
    return _client


def reset_client() -> None:
    """Drop cached client (for tests)."""
    global _client
    _client = None
