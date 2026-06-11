"""Vertex AI gemini-embedding-2 client for GraphRAG ontology and query vectors."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from google.genai import types
from google.genai.client import Client as GenaiClient
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from lib.genai.client import create_genai_client
from lib.genai.errors import is_resource_exhausted

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768
# gemini-embedding-2 is served from the global Vertex endpoint (not regional).
EMBEDDING_LOCATION = "global"
# Pace requests to stay under per-minute global_embed_content quotas on new projects.
_EMBED_REQUEST_INTERVAL_SEC = 1.5

_embedding_client: GenaiClient | None = None
_last_embed_request_at: float = 0.0


def _get_embedding_client(*, settings: Settings) -> GenaiClient:
    """Return a cached google-genai client for the global embedding endpoint."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = create_genai_client(settings=settings, location=EMBEDDING_LOCATION)
        logger.debug(
            "GenAI embedding client initialized for project=%s location=%s",
            settings.gcp_project_id,
            EMBEDDING_LOCATION,
        )
    return _embedding_client


def _throttle_embed_request() -> None:
    """Sleep between embed API calls to avoid bursting past per-minute Vertex quotas."""
    global _last_embed_request_at
    now = time.monotonic()
    elapsed = now - _last_embed_request_at
    if _last_embed_request_at > 0 and elapsed < _EMBED_REQUEST_INTERVAL_SEC:
        time.sleep(_EMBED_REQUEST_INTERVAL_SEC - elapsed)
    _last_embed_request_at = time.monotonic()


@retry(
    retry=retry_if_exception(is_resource_exhausted),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
)
def _embed_one_text_sync(
    *,
    client: GenaiClient,
    text: str,
    output_dimensionality: int,
) -> list[float]:
    """Embed a single text via gemini-embedding-2 with fixed output dimensionality."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
    )
    if not response.embeddings:
        msg = "embed_content returned no embeddings"
        raise ValueError(msg)
    values = response.embeddings[0].values
    if values is None:
        msg = "embed_content returned empty embedding values"
        raise ValueError(msg)
    return list(values)


def _embed_texts_sync(
    texts: Sequence[str],
    *,
    client: GenaiClient,
    output_dimensionality: int = EMBEDDING_DIMENSION,
) -> list[list[float]]:
    """Embed each text individually; Vertex gemini-embedding-2 accepts one content per call."""
    vectors: list[list[float]] = []
    for text in texts:
        _throttle_embed_request()
        vectors.append(
            _embed_one_text_sync(
                client=client,
                text=text,
                output_dimensionality=output_dimensionality,
            )
        )
    return vectors


async def embed_texts(
    texts: list[str],
    *,
    settings: Settings | None = None,
    client: GenaiClient | None = None,
) -> list[list[float]]:
    """Embed texts via gemini-embedding-2; returns one 768-dim vector per input."""
    if not texts:
        return []

    cfg = settings or get_settings()
    embedding_client = client or _get_embedding_client(settings=cfg)
    return await asyncio.to_thread(
        _embed_texts_sync,
        texts,
        client=embedding_client,
        output_dimensionality=EMBEDDING_DIMENSION,
    )
