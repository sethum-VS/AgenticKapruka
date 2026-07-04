"""Unit tests for NVIDIA NIM embeddings."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import openai

from app.config import Settings
from lib.embeddings.nvidia_embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    _embed_texts_sync,
    embed_texts,
)
from lib.redis.client import RedisClient


def _nim_settings(**overrides: object) -> Settings:
    base = {
        "redis_url": "redis://localhost:6379/0",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "secret",
        "zep_api_key": "zep-key",
        "nvidia_api_key": "nvidia-key",
        "session_secret": "x" * 32,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_embed_texts_empty_returns_empty() -> None:
    assert await embed_texts([]) == []


@pytest.mark.asyncio
async def test_embed_texts_fetches_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    redis_mock = MagicMock(spec=RedisClient)
    
    async def fake_get(*args, **kwargs) -> list[float]:
        return [0.1, 0.2]
        
    monkeypatch.setattr("lib.embeddings.nvidia_embeddings.get_cached_embedding", fake_get)
    
    result = await embed_texts(["test text"], redis_client=redis_mock)
    
    assert result == [[0.1, 0.2]]


def test_embed_texts_sync_rate_limit_retry() -> None:
    client_mock = MagicMock()
    
    # Fail 2 times with RateLimitError, succeed on 3rd
    class FakeResponse:
        @property
        def data(self):
            class Item:
                index = 0
                embedding = [0.9] * EMBEDDING_DIMENSION
            return [Item()]
            
    # OpenAI RateLimitError expects (message, response, body)
    response_mock = MagicMock()
    response_mock.status_code = 429
    
    error = openai.RateLimitError("Rate limit exceeded", response=response_mock, body=None)
    
    client_mock.embeddings.create.side_effect = [
        error,
        error,
        FakeResponse(),
    ]
    
    # Mock tenor to speed up
    with patch("lib.embeddings.nvidia_embeddings.wait_exponential", return_value=lambda *a, **k: 0.01):
        result = _embed_texts_sync(["retry me"], client=client_mock)
        
    assert client_mock.embeddings.create.call_count == 3
    assert len(result) == 1
    assert len(result[0]) == EMBEDDING_DIMENSION
