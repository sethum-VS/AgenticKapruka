"""Tests for batch ontology node embedding."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from lib.embeddings.vertex_embeddings import EMBEDDING_DIMENSION
from lib.kapruka.types import CategoryNode
from lib.neo4j.client import Neo4jClient
from lib.neo4j.embed_ontology import (
    build_embedding_text,
    clear_ontology_embeddings,
    count_nodes_with_embedding,
    embed_ontology_nodes,
    fetch_nodes_missing_embedding,
    has_category_embeddings,
    set_node_embeddings,
)
from lib.neo4j.ingest_categories import ingest_category_tree
from lib.neo4j.ontology import LABEL_CATEGORY, LABEL_OCCASION, LABEL_PRODUCT_TYPE

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
]


class _MockAsyncResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def data(self) -> list[dict[str, Any]]:
        return self._rows


class _EmbedMockStore:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.nodes: dict[str, set[str]] = defaultdict(set)
        self.node_properties: dict[str, dict[str, Any]] = {}

    def respond(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        batch_preview = str(parameters.get("batch", [{}])[0:1])
        if cypher.startswith("UNWIND $batch") and "occasion_id" in batch_preview:
            for row in parameters.get("batch", []):
                self._merge_triplet(row)
            return []

        if cypher.startswith("UNWIND $batch") and "embedding" in batch_preview:
            for row in parameters.get("batch", []):
                node_id = row["id"]
                if node_id in self.node_properties:
                    self.node_properties[node_id]["embedding"] = row["embedding"]
            return []

        if "REMOVE n.embedding" in cypher:
            cleared = 0
            for _node_id, props in self.node_properties.items():
                if props.get("embedding") is not None:
                    del props["embedding"]
                    cleared += 1
            return [{"cleared": cleared}]

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

        if "count(c) > 0 AS has_embeddings" in cypher:
            has_any = any(
                LABEL_CATEGORY in self.nodes
                and node_id in self.nodes[LABEL_CATEGORY]
                and self.node_properties.get(node_id, {}).get("embedding") is not None
                for node_id in self.node_properties
            )
            return [{"has_embeddings": has_any}]

        if "n.embedding IS NOT NULL" in cypher and "count(n) AS count" in cypher:
            label = parameters["label"]
            count = sum(
                1
                for node_id in self.nodes.get(label, set())
                if self.node_properties.get(node_id, {}).get("embedding") is not None
            )
            return [{"count": count}]

        return []

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
            LABEL_OCCASION,
            occ_id,
            {
                "slug": row["occasion_slug"],
                "display_name": row["occasion_display_name"],
                "description": row["occasion_description"],
                "kapruka_id": row["occasion_kapruka_id"],
            },
        )
        self._store_node(
            LABEL_PRODUCT_TYPE,
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
        self.node_properties[node_id] = {"id": node_id, "_label": label, **props}


class _EmbedMockSession:
    def __init__(self, store: _EmbedMockStore) -> None:
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

    async def __aenter__(self) -> _EmbedMockSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _EmbedMockDriver:
    def __init__(self, store: _EmbedMockStore) -> None:
        self._store = store

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncIterator[_EmbedMockSession]:
        del kwargs
        yield _EmbedMockSession(self._store)

    async def close(self) -> None:
        return None


def _client_with_store(store: _EmbedMockStore) -> Neo4jClient:
    return Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=_EmbedMockDriver(store),  # type: ignore[arg-type]
    )


def _fake_vector(seed: str) -> list[float]:
    return [float(len(seed) % 10) / 10.0] * EMBEDDING_DIMENSION


async def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [_fake_vector(text) for text in texts]


def test_build_embedding_text_combines_display_name_and_description() -> None:
    text = build_embedding_text(
        display_name="Birthday",
        description="Birthday — shop birthday gifts and products in Cakes.",
    )
    assert text.startswith("Birthday.")
    assert "shop birthday gifts" in text


def test_build_embedding_text_uses_display_name_only_when_no_description() -> None:
    assert build_embedding_text(display_name="Cakes", description=None) == "Cakes"


async def test_fetch_nodes_missing_embedding_returns_unembedded_nodes() -> None:
    store = _EmbedMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)

    pending = await fetch_nodes_missing_embedding(client)

    assert len(pending) == 3
    assert {node.label for node in pending} == {
        LABEL_CATEGORY,
        LABEL_OCCASION,
        LABEL_PRODUCT_TYPE,
    }
    await client.close()


async def test_set_node_embeddings_writes_vectors() -> None:
    store = _EmbedMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)
    vector = _fake_vector("cakes")

    await set_node_embeddings(client, [{"id": "category:cakes", "embedding": vector}])

    assert store.node_properties["category:cakes"]["embedding"] == vector
    await client.close()


async def test_embed_ontology_nodes_batches_and_sets_embeddings() -> None:
    store = _EmbedMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)

    stats = await embed_ontology_nodes(client, embed_fn=_fake_embed)

    assert stats.nodes_embedded == 3
    assert stats.batches_written == 1
    for node_id in store.node_properties:
        assert len(store.node_properties[node_id]["embedding"]) == EMBEDDING_DIMENSION
    await client.close()


async def test_embed_ontology_nodes_skips_when_all_embedded() -> None:
    store = _EmbedMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)
    await embed_ontology_nodes(client, embed_fn=_fake_embed)

    stats = await embed_ontology_nodes(client, embed_fn=_fake_embed)

    assert stats.nodes_embedded == 0
    assert stats.batches_written == 0
    await client.close()


async def test_has_category_embeddings_true_after_embed() -> None:
    store = _EmbedMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)
    await embed_ontology_nodes(client, embed_fn=_fake_embed)

    assert await has_category_embeddings(client) is True
    await client.close()


async def test_count_nodes_with_embedding_by_label() -> None:
    store = _EmbedMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)
    await embed_ontology_nodes(client, embed_fn=_fake_embed)

    assert await count_nodes_with_embedding(client, label=LABEL_CATEGORY) == 1
    assert await count_nodes_with_embedding(client, label=LABEL_OCCASION) == 1
    assert await count_nodes_with_embedding(client, label=LABEL_PRODUCT_TYPE) == 1
    await client.close()


async def test_embed_ontology_nodes_raises_on_vector_count_mismatch() -> None:
    store = _EmbedMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)

    async def bad_embed(texts: list[str]) -> list[list[float]]:
        return [_fake_vector("x")]  # too few vectors

    with pytest.raises(ValueError, match="embed_fn returned"):
        await embed_ontology_nodes(client, embed_fn=bad_embed)

    await client.close()


async def test_clear_ontology_embeddings_removes_existing_vectors() -> None:
    store = _EmbedMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)
    await embed_ontology_nodes(client, embed_fn=_fake_embed)

    cleared = await clear_ontology_embeddings(client)

    assert cleared == 3
    assert await has_category_embeddings(client) is False
    await client.close()
