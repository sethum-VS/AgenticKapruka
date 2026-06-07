"""Unit tests for Vertex AI text-embedding-005 embeddings client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from vertexai.language_models import TextEmbedding

from app.config import Settings
from lib.embeddings.vertex_embeddings import (
    EMBEDDING_DIMENSION,
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
        google_api_key="google-test-key",
        gcp_project_id="test-project",
        gcp_location="us-central1",
        session_secret="x" * 32,
    )


@pytest.mark.asyncio
async def test_embed_texts_returns_768_dim_vector() -> None:
    """Mocked Vertex model returns a 768-dimensional embedding per text."""
    mock_model = MagicMock()
    mock_model.get_embeddings.return_value = [
        TextEmbedding(values=[0.1] * EMBEDDING_DIMENSION),
    ]

    vectors = await embed_texts(["birthday cakes"], settings=_settings(), model=mock_model)

    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIMENSION
    mock_model.get_embeddings.assert_called_once_with(["birthday cakes"])


@pytest.mark.asyncio
async def test_embed_texts_batch_returns_one_vector_per_text() -> None:
    mock_model = MagicMock()
    mock_model.get_embeddings.return_value = [
        TextEmbedding(values=[0.1] * EMBEDDING_DIMENSION),
        TextEmbedding(values=[0.2] * EMBEDDING_DIMENSION),
    ]

    texts = ["birthday cakes", "wedding flowers"]
    vectors = await embed_texts(texts, settings=_settings(), model=mock_model)

    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)
    mock_model.get_embeddings.assert_called_once_with(texts)


@pytest.mark.asyncio
async def test_embed_texts_empty_list_skips_api_call() -> None:
    mock_model = MagicMock()

    vectors = await embed_texts([], settings=_settings(), model=mock_model)

    assert vectors == []
    mock_model.get_embeddings.assert_not_called()


@pytest.mark.asyncio
async def test_embed_texts_configures_vertex_from_settings() -> None:
    """When no model is injected, client initializes Vertex with Settings project/location."""
    mock_model = MagicMock()
    mock_model.get_embeddings.return_value = [
        TextEmbedding(values=[0.5] * EMBEDDING_DIMENSION),
    ]
    settings = _settings()

    with (
        patch("lib.embeddings.vertex_embeddings.vertexai.init") as mock_init,
        patch(
            "lib.embeddings.vertex_embeddings.TextEmbeddingModel.from_pretrained",
            return_value=mock_model,
        ) as mock_from_pretrained,
        patch("lib.embeddings.vertex_embeddings._vertex_initialized", False),
    ):
        vectors = await embed_texts(["gift for mom"], settings=settings)

    mock_init.assert_called_once_with(
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )
    mock_from_pretrained.assert_called_once_with(EMBEDDING_MODEL)
    assert len(vectors[0]) == EMBEDDING_DIMENSION
