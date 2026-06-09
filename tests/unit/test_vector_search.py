"""Tests for Neo4j Category vector index creation and similarity search."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from lib.embeddings.vertex_embeddings import EMBEDDING_DIMENSION
from lib.kapruka.types import CategoryNode
from lib.neo4j.client import Neo4jClient
from lib.neo4j.embed_ontology import build_embedding_text, embed_ontology_nodes
from lib.neo4j.ingest_categories import ingest_category_tree
from lib.neo4j.ontology import LABEL_CATEGORY, LABEL_OCCASION
from lib.neo4j.vector_search import (
    OCCASION_VECTOR_INDEX_NAME,
    VECTOR_INDEX_NAME,
    VECTOR_SIMILARITY_FUNCTION,
    create_category_vector_index,
    create_occasion_vector_index,
    create_ontology_vector_indexes,
    has_category_vector_index,
    has_occasion_vector_index,
    occasion_vector_search,
    vector_search,
)

_SAMPLE_TREE: list[CategoryNode] = [
    CategoryNode(
        name="Cakes",
        url="https://www.kapruka.com/online/cakes",
        children=[
            CategoryNode(
                name="Birthday",
                url="https://www.kapruka.com/online/cakes/birthday",
                children=[],
            ),
        ],
    ),
    CategoryNode(
        name="Flowers",
        url="https://www.kapruka.com/online/flowers",
        children=[],
    ),
]

_KEYWORD_INDICES: dict[str, list[int]] = {
    "birthday": [0, 1, 2, 3],
    "gift": [1, 2, 3, 4],
    "cake": [0, 2, 4, 5],
    "cakes": [0, 2, 4, 5],
    "flower": [10, 11, 12, 13],
    "flowers": [10, 11, 12, 13],
    "wedding": [10, 11, 14, 15],
}


class _MockAsyncResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def data(self) -> list[dict[str, Any]]:
        return self._rows


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _semantic_fake_vector(text: str) -> list[float]:
    """Deterministic keyword-overlap vectors for semantic-ish similarity in tests."""
    vec = [0.0] * EMBEDDING_DIMENSION
    text_lower = text.lower()
    for keyword, indices in _KEYWORD_INDICES.items():
        if keyword in text_lower:
            for index in indices:
                vec[index] += 1.0
    if not any(vec):
        vec[0] = 1.0
    norm = math.sqrt(sum(value * value for value in vec))
    return [value / norm for value in vec]


async def _semantic_fake_embed(texts: list[str]) -> list[list[float]]:
    return [_semantic_fake_vector(text) for text in texts]


class _VectorSearchMockStore:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.nodes: dict[str, set[str]] = defaultdict(set)
        self.node_properties: dict[str, dict[str, Any]] = {}
        self.vector_indexes: dict[str, dict[str, Any]] = {}

    def respond(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        batch_preview = str(parameters.get("batch", [{}])[0:1])
        if cypher.startswith("UNWIND $batch") and "occasion_id" in batch_preview:
            for row in parameters.get("batch", []):
                self._merge_triplet(row)
            return []

        if cypher.startswith("UNWIND $batch") and "category_id" in batch_preview:
            for row in parameters.get("batch", []):
                self._merge_category(row)
            return []

        if cypher.startswith("UNWIND $batch") and "embedding" in batch_preview:
            for row in parameters.get("batch", []):
                node_id = row["id"]
                if node_id in self.node_properties:
                    self.node_properties[node_id]["embedding"] = row["embedding"]
            return []

        if "n.embedding IS NULL" in cypher:
            rows = []
            for node_id, props in sorted(self.node_properties.items()):
                if props.get("embedding") is not None:
                    continue
                if props.get("display_name") is None:
                    continue
                label = props.get("_label", LABEL_CATEGORY)
                rows.append(
                    {
                        "id": node_id,
                        "label": label,
                        "display_name": props["display_name"],
                        "description": props.get("description"),
                    }
                )
            return rows

        if cypher.startswith("CREATE VECTOR INDEX"):
            name = cypher.split()[3]
            label = LABEL_CATEGORY
            if f"FOR (n:{LABEL_OCCASION})" in cypher:
                label = LABEL_OCCASION
            elif f"FOR (n:{LABEL_CATEGORY})" in cypher:
                label = LABEL_CATEGORY
            self.vector_indexes[name] = {
                "name": name,
                "type": "VECTOR",
                "labelsOrTypes": [label],
                "properties": ["embedding"],
                "options": {
                    "indexConfig": {
                        "`vector.dimensions`": EMBEDDING_DIMENSION,
                        "`vector.similarity_function`": VECTOR_SIMILARITY_FUNCTION,
                    }
                },
            }
            return []

        if cypher.startswith("SHOW INDEXES"):
            name = parameters.get("name")
            if name and name in self.vector_indexes:
                return [self.vector_indexes[name]]
            return []

        if cypher.startswith("CALL db.index.vector.queryNodes"):
            query_embedding = parameters["query_embedding"]
            top_k = int(parameters["top_k"])
            index_name = parameters["index_name"]
            label = LABEL_CATEGORY
            if index_name == OCCASION_VECTOR_INDEX_NAME:
                label = LABEL_OCCASION
            hits: list[dict[str, Any]] = []
            for node_id in self.nodes.get(label, set()):
                props = self.node_properties.get(node_id, {})
                embedding = props.get("embedding")
                if embedding is None:
                    continue
                score = _cosine_similarity(query_embedding, embedding)
                hits.append({"id": node_id, "score": score, "_node": props})
            hits.sort(key=lambda row: row["score"], reverse=True)
            return [{"id": row["id"], "score": row["score"]} for row in hits[:top_k]]

        return []

    def _merge_category(self, row: dict[str, Any]) -> None:
        cat_id = row["category_id"]
        self._store_node(
            LABEL_CATEGORY,
            cat_id,
            {
                "slug": row["slug"],
                "display_name": row["display_name"],
                "description": row["description"],
                "kapruka_id": row["kapruka_id"],
            },
        )

    def _merge_triplet(self, row: dict[str, Any]) -> None:
        cat_id = row["category_id"]
        occ_id = row["occasion_id"]
        pt_id = row["product_type_id"]
        self._store_node(
            LABEL_CATEGORY,
            cat_id,
            {
                "slug": row["category_slug"],
                "display_name": row["category_display_name"],
                "description": row["category_description"],
                "kapruka_id": row["category_kapruka_id"],
            },
        )
        self._store_node(
            "Occasion",
            occ_id,
            {
                "slug": row["occasion_slug"],
                "display_name": row["occasion_display_name"],
                "description": row["occasion_description"],
                "kapruka_id": row["occasion_kapruka_id"],
            },
        )
        self._store_node(
            "ProductType",
            pt_id,
            {
                "slug": row["product_type_slug"],
                "display_name": row["product_type_display_name"],
                "description": row["product_type_description"],
                "kapruka_id": row["product_type_kapruka_id"],
            },
        )

    def _store_node(self, label: str, node_id: str, props: dict[str, Any]) -> None:
        self.nodes[label].add(node_id)
        existing = self.node_properties.setdefault(node_id, {})
        existing.update(props)
        existing["_label"] = label
        existing["id"] = node_id


class _VectorSearchMockSession:
    def __init__(self, store: _VectorSearchMockStore) -> None:
        self._store = store

    async def run(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _MockAsyncResult:
        del kwargs
        self._store.executed.append((cypher.strip(), parameters or {}))
        return _MockAsyncResult(self._store.respond(cypher.strip(), parameters or {}))

    async def __aenter__(self) -> _VectorSearchMockSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _VectorSearchMockDriver:
    def __init__(self, store: _VectorSearchMockStore) -> None:
        self._store = store

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncIterator[_VectorSearchMockSession]:
        del kwargs
        yield _VectorSearchMockSession(self._store)

    async def close(self) -> None:
        return None


def _client_with_store(store: _VectorSearchMockStore) -> Neo4jClient:
    return Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=_VectorSearchMockDriver(store),  # type: ignore[arg-type]
    )


async def test_create_category_vector_index_executes_create_statement() -> None:
    store = _VectorSearchMockStore()
    client = _client_with_store(store)

    await create_category_vector_index(client)

    assert any(
        cypher.startswith("CREATE VECTOR INDEX") and VECTOR_INDEX_NAME in cypher
        for cypher, _ in store.executed
    )
    assert await has_category_vector_index(client) is True
    await client.close()


async def test_vector_search_returns_ids_and_scores() -> None:
    store = _VectorSearchMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)
    await embed_ontology_nodes(client, embed_fn=_semantic_fake_embed)
    await create_category_vector_index(client)

    query = _semantic_fake_vector("birthday gift cakes")
    hits = await vector_search(client, query, top_k=2)

    assert len(hits) >= 1
    assert all(hit.id.startswith("category:") for hit in hits)
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)
    if len(hits) > 1:
        assert hits[0].score >= hits[1].score
    await client.close()


async def test_vector_search_birthday_gift_prefers_cakes_category() -> None:
    """Integration-style test: 'birthday gift' should rank Cakes above Flowers."""
    store = _VectorSearchMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)
    await embed_ontology_nodes(client, embed_fn=_semantic_fake_embed)
    await create_category_vector_index(client)

    query_text = build_embedding_text(
        display_name="birthday gift",
        description="Looking for a birthday gift",
    )
    query_embedding = (await _semantic_fake_embed([query_text]))[0]
    hits = await vector_search(client, query_embedding, top_k=5)

    assert hits
    assert hits[0].id == "category:cakes"
    hit_ids = {hit.id for hit in hits}
    assert "category:flowers" in hit_ids
    cakes_score = next(hit.score for hit in hits if hit.id == "category:cakes")
    flowers_score = next(hit.score for hit in hits if hit.id == "category:flowers")
    assert cakes_score > flowers_score
    await client.close()


async def test_vector_search_rejects_wrong_embedding_dimension() -> None:
    store = _VectorSearchMockStore()
    client = _client_with_store(store)

    with pytest.raises(ValueError, match="768 dimensions"):
        await vector_search(client, [0.1, 0.2, 0.3])

    await client.close()


async def test_vector_search_rejects_invalid_top_k() -> None:
    store = _VectorSearchMockStore()
    client = _client_with_store(store)
    query = _semantic_fake_vector("birthday gift")

    with pytest.raises(ValueError, match="top_k"):
        await vector_search(client, query, top_k=0)

    await client.close()


async def test_create_occasion_vector_index_executes_create_statement() -> None:
    store = _VectorSearchMockStore()
    client = _client_with_store(store)

    await create_occasion_vector_index(client)

    create_cyphers = [
        cypher for cypher, _ in store.executed if cypher.startswith("CREATE VECTOR INDEX")
    ]
    assert len(create_cyphers) == 1
    assert OCCASION_VECTOR_INDEX_NAME in create_cyphers[0]
    assert f"FOR (n:{LABEL_OCCASION})" in create_cyphers[0]
    assert await has_occasion_vector_index(client) is True
    await client.close()


async def test_create_ontology_vector_indexes_creates_category_then_occasion() -> None:
    store = _VectorSearchMockStore()
    client = _client_with_store(store)

    await create_ontology_vector_indexes(client)

    create_cyphers = [
        cypher for cypher, _ in store.executed if cypher.startswith("CREATE VECTOR INDEX")
    ]
    assert len(create_cyphers) == 2
    assert VECTOR_INDEX_NAME in create_cyphers[0]
    assert f"FOR (n:{LABEL_CATEGORY})" in create_cyphers[0]
    assert OCCASION_VECTOR_INDEX_NAME in create_cyphers[1]
    assert f"FOR (n:{LABEL_OCCASION})" in create_cyphers[1]
    assert await has_category_vector_index(client) is True
    assert await has_occasion_vector_index(client) is True
    await client.close()


async def test_occasion_vector_search_returns_ids_and_scores() -> None:
    store = _VectorSearchMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)
    await embed_ontology_nodes(client, embed_fn=_semantic_fake_embed)
    await create_occasion_vector_index(client)

    query = _semantic_fake_vector("birthday gift")
    hits = await occasion_vector_search(client, query, top_k=2)

    assert len(hits) >= 1
    assert all(hit.id.startswith("occasion:") for hit in hits)
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)
    if len(hits) > 1:
        assert hits[0].score >= hits[1].score

    query_calls = [
        params
        for cypher, params in store.executed
        if cypher.startswith("CALL db.index.vector.queryNodes")
    ]
    assert query_calls[-1]["index_name"] == OCCASION_VECTOR_INDEX_NAME
    await client.close()


async def test_occasion_vector_search_rejects_wrong_embedding_dimension() -> None:
    store = _VectorSearchMockStore()
    client = _client_with_store(store)

    with pytest.raises(ValueError, match="768 dimensions"):
        await occasion_vector_search(client, [0.1, 0.2, 0.3])

    await client.close()
