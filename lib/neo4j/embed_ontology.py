"""Batch-embed ontology nodes (Occasion, Category, ProductType) via Vertex AI."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from lib.embeddings.vertex_embeddings import embed_texts
from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import (
    LABEL_CATEGORY,
    LABEL_OCCASION,
    LABEL_PRODUCT_TYPE,
)

_EMBED_BATCH_SIZE: Final = 50

_ONTOLOGY_LABELS: Final = (LABEL_OCCASION, LABEL_CATEGORY, LABEL_PRODUCT_TYPE)

_FETCH_MISSING_EMBEDDING_CYPHER = """
MATCH (n)
WHERE any(l IN labels(n) WHERE l IN $ontology_labels)
  AND n.embedding IS NULL
  AND n.display_name IS NOT NULL
RETURN n.id AS id,
       labels(n)[0] AS label,
       n.display_name AS display_name,
       n.description AS description
ORDER BY n.id
""".strip()

_SET_EMBEDDINGS_BATCH_CYPHER = """
UNWIND $batch AS row
MATCH (n {id: row.id})
WHERE any(l IN labels(n) WHERE l IN $ontology_labels)
SET n.embedding = row.embedding
""".strip()

_COUNT_WITH_EMBEDDING_CYPHER = """
MATCH (n)
WHERE $label IN labels(n) AND n.embedding IS NOT NULL
RETURN count(n) AS count
""".strip()

_HAS_CATEGORY_EMBEDDINGS_CYPHER = """
MATCH (c:Category) WHERE c.embedding IS NOT NULL RETURN count(c) > 0 AS has_embeddings
""".strip()

EmbedTextsFn = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass(frozen=True, slots=True)
class OntologyNodeForEmbedding:
    """Ontology node row returned when embedding is missing."""

    id: str
    label: str
    display_name: str
    description: str | None


@dataclass(frozen=True, slots=True)
class EmbedOntologyStats:
    """Summary counts from a batch ontology embedding run."""

    nodes_embedded: int
    batches_written: int


def build_embedding_text(*, display_name: str, description: str | None) -> str:
    """Combine display_name and description for Vertex text-embedding-005."""
    name = display_name.strip()
    if description and description.strip():
        return f"{name}. {description.strip()}"
    return name


async def fetch_nodes_missing_embedding(
    client: Neo4jClient,
) -> list[OntologyNodeForEmbedding]:
    """Return ontology nodes that lack an embedding but have display_name."""
    rows = await client.execute(
        _FETCH_MISSING_EMBEDDING_CYPHER,
        {"ontology_labels": list(_ONTOLOGY_LABELS)},
    )
    return [
        OntologyNodeForEmbedding(
            id=str(row["id"]),
            label=str(row["label"]),
            display_name=str(row["display_name"]),
            description=str(row["description"]) if row.get("description") is not None else None,
        )
        for row in rows
    ]


def _chunked(
    items: Sequence[OntologyNodeForEmbedding],
    size: int,
) -> list[list[OntologyNodeForEmbedding]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


async def set_node_embeddings(
    client: Neo4jClient,
    updates: list[dict[str, Any]],
) -> None:
    """SET n.embedding for a batch of ontology nodes keyed by id."""
    if not updates:
        return
    await client.execute(
        _SET_EMBEDDINGS_BATCH_CYPHER,
        {"batch": updates, "ontology_labels": list(_ONTOLOGY_LABELS)},
    )


async def embed_ontology_nodes(
    client: Neo4jClient,
    *,
    embed_fn: EmbedTextsFn | None = None,
    batch_size: int = _EMBED_BATCH_SIZE,
) -> EmbedOntologyStats:
    """Embed display_name+description for all ontology nodes missing embedding."""
    embed = embed_fn or embed_texts
    pending = await fetch_nodes_missing_embedding(client)
    if not pending:
        return EmbedOntologyStats(nodes_embedded=0, batches_written=0)

    nodes_embedded = 0
    batches_written = 0

    for batch in _chunked(pending, batch_size):
        texts = [
            build_embedding_text(
                display_name=node.display_name,
                description=node.description,
            )
            for node in batch
        ]
        vectors = await embed(texts)
        if len(vectors) != len(batch):
            msg = f"embed_fn returned {len(vectors)} vectors for {len(batch)} texts"
            raise ValueError(msg)

        updates = [
            {"id": node.id, "embedding": vector}
            for node, vector in zip(batch, vectors, strict=True)
        ]
        await set_node_embeddings(client, updates)
        nodes_embedded += len(batch)
        batches_written += 1

    return EmbedOntologyStats(
        nodes_embedded=nodes_embedded,
        batches_written=batches_written,
    )


async def count_nodes_with_embedding(client: Neo4jClient, *, label: str) -> int:
    """Return how many nodes of the given label have a non-null embedding."""
    rows = await client.execute(_COUNT_WITH_EMBEDDING_CYPHER, {"label": label})
    return int(rows[0]["count"]) if rows else 0


async def has_category_embeddings(client: Neo4jClient) -> bool:
    """Return True when at least one Category node has an embedding (PRD-045 verify)."""
    rows = await client.execute(_HAS_CATEGORY_EMBEDDINGS_CYPHER)
    return bool(rows and rows[0].get("has_embeddings"))
