"""Redis cache for Vertex query embeddings (GraphRAG)."""

from __future__ import annotations

import hashlib
import json
from typing import Final

from lib.redis.client import RedisClient

EMBEDDING_CACHE_TTL: Final = 3600
EMBEDDING_CACHE_PREFIX: Final = "embed:"


def embedding_cache_key(text: str) -> str:
    """Build Redis key embed:sha256(normalized_text)."""
    digest = hashlib.sha256(text.strip().encode()).hexdigest()
    return f"{EMBEDDING_CACHE_PREFIX}{digest}"


async def get_cached_embedding(
    redis_client: RedisClient,
    text: str,
) -> list[float] | None:
    """Return a cached embedding vector or None on miss."""
    key = embedding_cache_key(text)
    raw = await redis_client.client.get(key)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    return [float(value) for value in payload]


async def set_cached_embedding(
    redis_client: RedisClient,
    text: str,
    vector: list[float],
    *,
    ttl: int = EMBEDDING_CACHE_TTL,
) -> None:
    """Store an embedding vector with TTL (default 1 hour)."""
    key = embedding_cache_key(text)
    await redis_client.client.set(key, json.dumps(vector), ex=ttl)
