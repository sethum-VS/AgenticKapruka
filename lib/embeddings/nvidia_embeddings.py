"""NVIDIA NIM openai-compatible client for GraphRAG ontology and query vectors."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from lib.embeddings.embedding_cache import get_cached_embedding, set_cached_embedding
from lib.genai.client import create_nvidia_client
from lib.genai.rate_limiter import get_rate_limiter
from lib.redis.client import RedisClient

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nvidia/nv-embed-v1"
EMBEDDING_DIMENSION = 4096


def _is_rate_limit(exc: BaseException) -> bool:
    """True if 429 rate limit error."""
    return isinstance(exc, openai.RateLimitError)


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
)
def _embed_texts_sync(
    texts: Sequence[str],
    *,
    client: OpenAI,
) -> list[list[float]]:
    """Embed texts via nvidia/nv-embed-v1; returns 4096-dim vectors."""
    response = client.embeddings.create(
        input=list(texts),
        model=EMBEDDING_MODEL,
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    if not response.data:
        msg = "NVIDIA NIM returned no embeddings"
        raise ValueError(msg)
    
    # Sort data by index to ensure ordering matches the input array
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


async def embed_texts(
    texts: list[str],
    *,
    settings: Settings | None = None,
    client: OpenAI | None = None,
    redis_client: RedisClient | None = None,
) -> list[list[float]]:
    """Embed texts via nvidia/nv-embed-v1; returns one 4096-dim vector per input."""
    if not texts:
        return []

    if redis_client is not None and len(texts) == 1:
        stripped = texts[0].strip()
        if stripped:
            cached = await get_cached_embedding(redis_client, stripped)
            if cached is not None:
                return [cached]

    cfg = settings or get_settings()
    embedding_client = client or create_nvidia_client(settings=cfg)
    await get_rate_limiter().acquire()

    vectors = await asyncio.to_thread(
        _embed_texts_sync,
        texts,
        client=embedding_client,
    )

    if redis_client is not None and len(texts) == 1 and vectors:
        stripped = texts[0].strip()
        if stripped:
            await set_cached_embedding(redis_client, stripped, vectors[0])

    return vectors
