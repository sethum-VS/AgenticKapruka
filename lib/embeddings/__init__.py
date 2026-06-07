"""Text embedding clients for GraphRAG."""

from lib.embeddings.vertex_embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    embed_texts,
)

__all__ = [
    "EMBEDDING_DIMENSION",
    "EMBEDDING_MODEL",
    "embed_texts",
]
