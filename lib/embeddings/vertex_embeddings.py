"""Vertex AI text-embedding-005 client for GraphRAG ontology and query vectors."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import vertexai
from vertexai.language_models import TextEmbeddingModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-005"
EMBEDDING_DIMENSION = 768

_vertex_initialized = False


def _ensure_vertex_init(*, project_id: str, location: str) -> None:
    """Initialize Vertex AI once per process with project and region from Settings."""
    global _vertex_initialized
    if _vertex_initialized:
        return
    vertexai.init(project=project_id, location=location)
    _vertex_initialized = True
    logger.debug("Vertex AI initialized for project=%s location=%s", project_id, location)


def _load_embedding_model(*, project_id: str, location: str) -> TextEmbeddingModel:
    _ensure_vertex_init(project_id=project_id, location=location)
    return TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)


def _embed_texts_sync(texts: Sequence[str], *, model: TextEmbeddingModel) -> list[list[float]]:
    embeddings = model.get_embeddings(list(texts))
    return [list(embedding.values) for embedding in embeddings]


async def embed_texts(
    texts: list[str],
    *,
    settings: Settings | None = None,
    model: TextEmbeddingModel | None = None,
) -> list[list[float]]:
    """Embed texts via Vertex text-embedding-005; returns one 768-dim vector per input."""
    if not texts:
        return []

    cfg = settings or get_settings()
    embedding_model = model or _load_embedding_model(
        project_id=cfg.gcp_project_id,
        location=cfg.gcp_location,
    )
    return await asyncio.to_thread(_embed_texts_sync, texts, model=embedding_model)
