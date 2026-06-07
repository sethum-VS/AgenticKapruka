"""Neo4j vector index and similarity search over Category embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from lib.embeddings.vertex_embeddings import EMBEDDING_DIMENSION
from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import LABEL_CATEGORY

VECTOR_INDEX_NAME: Final = "ontology_category_embedding"
VECTOR_SIMILARITY_FUNCTION: Final = "cosine"

_CREATE_VECTOR_INDEX_CYPHER = f"""
CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
FOR (n:{LABEL_CATEGORY})
ON (n.embedding)
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {EMBEDDING_DIMENSION},
    `vector.similarity_function`: '{VECTOR_SIMILARITY_FUNCTION}'
  }}
}}
""".strip()

_VECTOR_SEARCH_CYPHER = """
CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
YIELD node, score
RETURN node.id AS id, score
ORDER BY score DESC
""".strip()

_SHOW_VECTOR_INDEX_CYPHER = """
SHOW INDEXES
YIELD name, type, labelsOrTypes, properties, options
WHERE name = $name
RETURN name, type, labelsOrTypes, properties, options
""".strip()


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """Single vector search result: ontology node id and similarity score."""

    id: str
    score: float


async def create_category_vector_index(client: Neo4jClient) -> None:
    """Create the ontology Category embedding vector index if it does not exist."""
    await client.execute(_CREATE_VECTOR_INDEX_CYPHER)


async def has_category_vector_index(client: Neo4jClient) -> bool:
    """Return True when the ontology Category vector index exists."""
    rows = await client.execute(_SHOW_VECTOR_INDEX_CYPHER, {"name": VECTOR_INDEX_NAME})
    return bool(rows)


async def list_category_vector_index(client: Neo4jClient) -> list[dict[str, Any]]:
    """Return SHOW INDEXES rows for the ontology Category vector index."""
    return await client.execute(_SHOW_VECTOR_INDEX_CYPHER, {"name": VECTOR_INDEX_NAME})


async def vector_search(
    client: Neo4jClient,
    query_embedding: list[float],
    *,
    top_k: int = 5,
) -> list[VectorSearchHit]:
    """Query Category nodes by embedding similarity; returns ids and scores (higher is better)."""
    if len(query_embedding) != EMBEDDING_DIMENSION:
        msg = (
            f"query_embedding must have {EMBEDDING_DIMENSION} dimensions, "
            f"got {len(query_embedding)}"
        )
        raise ValueError(msg)
    if top_k < 1:
        msg = "top_k must be >= 1"
        raise ValueError(msg)

    rows = await client.execute(
        _VECTOR_SEARCH_CYPHER,
        {
            "index_name": VECTOR_INDEX_NAME,
            "top_k": top_k,
            "query_embedding": query_embedding,
        },
    )
    return [VectorSearchHit(id=str(row["id"]), score=float(row["score"])) for row in rows]
