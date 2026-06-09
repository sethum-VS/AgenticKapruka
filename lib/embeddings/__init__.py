"""Text embedding clients for GraphRAG."""

from lib.embeddings.reranker import RERANKER_MODEL, CrossEncoderService, get_reranker
from lib.embeddings.vertex_embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    embed_texts,
)

__all__ = [
    "CrossEncoderService",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "embed_texts",
    "get_reranker",
]
