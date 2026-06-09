"""Unit tests for Vertex AI gemini-embedding-2 embeddings client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors

from app.config import Settings
from lib.embeddings.vertex_embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_LOCATION,
    EMBEDDING_MODEL,
    embed_texts,
)


def _settings() -> Settings:
    return Settings(
        redis_url="redis://localhost:6379/0",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test-password",
        zep_api_key="zep-test-key",
        gcp_project_id="test-project",
        gcp_location="us-central1",
        session_secret="x" * 32,
        _env_file=None,
    )


def _mock_embedding(values: list[float] | None = None) -> MagicMock:
    embedding = MagicMock()
    embedding.values = values or [0.1] * EMBEDDING_DIMENSION
    return embedding


def _mock_client(*, embeddings: list[MagicMock] | None = None) -> MagicMock:
    client = MagicMock()
    if embeddings is None:
        embeddings = [_mock_embedding()]
    client.models.embed_content.side_effect = [
        MagicMock(embeddings=[embedding]) for embedding in embeddings
    ]
    return client


@pytest.mark.asyncio
async def test_embed_texts_returns_768_dim_vector() -> None:
    """Mocked GenAI client returns a 768-dimensional embedding per text."""
    mock_client = _mock_client()

    vectors = await embed_texts(["birthday cakes"], settings=_settings(), client=mock_client)

    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIMENSION
    mock_client.models.embed_content.assert_called_once()
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == EMBEDDING_MODEL
    assert call_kwargs["contents"] == "birthday cakes"
    assert call_kwargs["config"].output_dimensionality == EMBEDDING_DIMENSION


@pytest.mark.asyncio
async def test_embed_texts_batch_returns_one_vector_per_text() -> None:
    mock_client = _mock_client(
        embeddings=[
            _mock_embedding([0.1] * EMBEDDING_DIMENSION),
            _mock_embedding([0.2] * EMBEDDING_DIMENSION),
        ]
    )

    texts = ["birthday cakes", "wedding flowers"]
    vectors = await embed_texts(texts, settings=_settings(), client=mock_client)

    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)
    assert mock_client.models.embed_content.call_count == 2


@pytest.mark.asyncio
async def test_embed_texts_empty_list_skips_api_call() -> None:
    mock_client = _mock_client()

    vectors = await embed_texts([], settings=_settings(), client=mock_client)

    assert vectors == []
    mock_client.models.embed_content.assert_not_called()


@pytest.mark.asyncio
async def test_embed_texts_configures_global_vertex_client_from_settings() -> None:
    """When no client is injected, embeddings use the global gemini-embedding-2 endpoint."""
    mock_client = _mock_client()
    settings = _settings()

    with (
        patch(
            "lib.embeddings.vertex_embeddings.create_genai_client",
            return_value=mock_client,
        ) as mock_create,
        patch("lib.embeddings.vertex_embeddings._embedding_client", None),
    ):
        vectors = await embed_texts(["gift for mom"], settings=settings)

    mock_create.assert_called_once_with(settings=settings, location=EMBEDDING_LOCATION)
    assert len(vectors[0]) == EMBEDDING_DIMENSION


@pytest.mark.asyncio
async def test_embed_texts_retries_on_resource_exhausted() -> None:
    from tenacity import wait_none

    import lib.embeddings.vertex_embeddings as vertex_module

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = [
        genai_errors.ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}}, None),
        MagicMock(embeddings=[_mock_embedding()]),
    ]

    with patch.object(vertex_module._embed_one_text_sync.retry, "wait", wait_none()):
        vectors = await embed_texts(["retry me"], settings=_settings(), client=mock_client)

    assert len(vectors) == 1
    assert mock_client.models.embed_content.call_count == 2


@pytest.mark.asyncio
async def test_embed_texts_retries_on_google_api_core_resource_exhausted() -> None:
    from tenacity import wait_none

    import lib.embeddings.vertex_embeddings as vertex_module

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = [
        google_exceptions.ResourceExhausted("quota"),
        MagicMock(embeddings=[_mock_embedding()]),
    ]

    with patch.object(vertex_module._embed_one_text_sync.retry, "wait", wait_none()):
        vectors = await embed_texts(["retry me"], settings=_settings(), client=mock_client)

    assert len(vectors) == 1
    assert mock_client.models.embed_content.call_count == 2
