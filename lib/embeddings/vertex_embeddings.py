"""Vertex AI gemini-embedding-2 client for GraphRAG ontology and query vectors."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors
from google.genai import types
from google.genai.client import Client as GenaiClient
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from lib.genai.client import create_genai_client

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768
# gemini-embedding-2 is served from the global Vertex endpoint (not regional).
EMBEDDING_LOCATION = "global"

_embedding_client: GenaiClient | None = None


def _is_resource_exhausted(exc: BaseException) -> bool:
    """Return True for Vertex/Gemini 429 RESOURCE_EXHAUSTED errors."""
    if isinstance(exc, google_exceptions.ResourceExhausted):
        return True
    if isinstance(exc, genai_errors.ClientError):
        if exc.code == 429:
            return True
        if exc.status == "RESOURCE_EXHAUSTED":
            return True
    return False


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


@retry(
    retry=retry_if_exception(_is_resource_exhausted),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
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
    return [
        _embed_one_text_sync(
            client=client,
            text=text,
            output_dimensionality=output_dimensionality,
        )
        for text in texts
    ]


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
