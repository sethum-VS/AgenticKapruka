"""Unit tests for query embedding Redis cache."""

from __future__ import annotations

import json

import fakeredis.aioredis
import pytest

from lib.embeddings.embedding_cache import (
    EMBEDDING_CACHE_TTL,
    embedding_cache_key,
    get_cached_embedding,
    set_cached_embedding,
)
from lib.embeddings.vertex_embeddings import embed_texts
from lib.redis.client import RedisClient


@pytest.mark.asyncio
async def test_embedding_cache_round_trip() -> None:
    fake = fakeredis.aioredis.FakeRedis()
    redis_client = RedisClient("redis://localhost", client=fake)
    vector = [0.1, 0.2, 0.3]

    await set_cached_embedding(redis_client, "birthday cake", vector)
    cached = await get_cached_embedding(redis_client, "birthday cake")

    assert cached == vector
    key = embedding_cache_key("birthday cake")
    raw = await fake.get(key)
    assert raw is not None
    assert json.loads(raw) == vector


@pytest.mark.asyncio
async def test_embed_texts_uses_redis_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fakeredis.aioredis.FakeRedis()
    redis_client = RedisClient("redis://localhost", client=fake)
    expected = [0.5] * 768
    embed_calls: list[list[str]] = []

    def fake_embed_sync(texts: list[str], **kwargs: object) -> list[list[float]]:
        embed_calls.append(list(texts))
        return [expected]

    async def fake_to_thread(fn: object, *args: object, **kwargs: object) -> object:
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        "lib.embeddings.vertex_embeddings._embed_texts_sync",
        fake_embed_sync,
    )
    monkeypatch.setattr(
        "lib.embeddings.vertex_embeddings.asyncio.to_thread",
        fake_to_thread,
    )

    first = await embed_texts(["birthday cake"], redis_client=redis_client)
    second = await embed_texts(["birthday cake"], redis_client=redis_client)

    assert first == [expected]
    assert second == [expected]
    assert len(embed_calls) == 1
