"""Multi-region Vertex AI failover for Gemini generate_content calls."""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types

from app.config import Settings, get_settings
from lib.genai.client import create_genai_client
from lib.genai.errors import is_resource_exhausted

logger = logging.getLogger(__name__)

DEFAULT_VERTEX_FALLBACK_REGIONS: tuple[str, ...] = (
    "europe-west4",
    "us-east4",
    "asia-northeast1",
    "us-central1",
)

_client_cache: dict[str, genai.Client] = {}


def vertex_location_chain(settings: Settings) -> list[str]:
    """Primary chat location followed by deduplicated fallback regions."""
    primary = settings.gemini_chat_location.strip()
    chain = [primary] if primary else []
    for region in settings.gemini_fallback_regions:
        stripped = region.strip()
        if stripped and stripped not in chain:
            chain.append(stripped)
    return chain


def _client_cache_key(settings: Settings, location: str) -> str:
    return f"{settings.gcp_project_id}:{location}"


def _get_client_for_location(settings: Settings, location: str) -> genai.Client:
    key = _client_cache_key(settings, location)
    cached = _client_cache.get(key)
    if cached is not None:
        return cached
    client = create_genai_client(settings=settings, location=location)
    _client_cache[key] = client
    return client


def clear_client_cache() -> None:
    """Drop cached regional clients (for tests)."""
    _client_cache.clear()


def generate_content_with_fallback(
    *,
    model: str,
    contents: Any,
    config: types.GenerateContentConfig,
    settings: Settings | None = None,
    client: genai.Client | None = None,
) -> types.GenerateContentResponse:
    """Call generate_content with multi-region failover on 429.

    When *client* is provided (tests/DI), uses that client only — no region cascade.
    For Vertex, tries ``gemini_chat_location`` then ``gemini_fallback_regions``.
    """
    if client is not None:
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    cfg = settings or get_settings()
    if cfg.gemini_backend == "api_key":
        api_client = create_genai_client(settings=cfg)
        return api_client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    locations = vertex_location_chain(cfg)
    last_exc: BaseException | None = None
    for idx, location in enumerate(locations):
        try:
            regional_client = _get_client_for_location(cfg, location)
            response = regional_client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            if idx > 0:
                logger.warning(
                    "Gemini generate_content succeeded via fallback region %s",
                    location,
                )
            return response
        except Exception as exc:
            if not is_resource_exhausted(exc):
                raise
            last_exc = exc
            if idx < len(locations) - 1:
                logger.warning(
                    "Gemini generate_content rate limited in %s; trying next region",
                    location,
                )
            else:
                logger.warning(
                    "Gemini generate_content rate limited in %s; no more regions",
                    location,
                )

    assert last_exc is not None
    raise last_exc
