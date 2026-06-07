"""Integration tests for HybridRAG retrieve_hybrid_context with Neo4j graph."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from graphs.nodes.retrieve_hybrid_context import retrieve_hybrid_context
from graphs.state import AgentState
from lib.embeddings.vertex_embeddings import EMBEDDING_DIMENSION
from lib.kapruka.types import CategoryNode
from lib.neo4j.client import Neo4jClient
from lib.neo4j.embed_ontology import embed_ontology_nodes
from lib.neo4j.ingest_categories import ingest_category_tree
from lib.neo4j.ontology import (
    LABEL_CATEGORY,
    LABEL_OCCASION,
    LABEL_PRODUCT_TYPE,
    REL_CATEGORY_TO_PRODUCT_TYPE,
    REL_OCCASION_TO_CATEGORY,
)
from lib.neo4j.vector_search import create_category_vector_index

_WEDDING_FLOWERS_TREE: list[CategoryNode] = [
    CategoryNode(
        name="Flowers",
        url="https://www.kapruka.com/online/flowers",
        children=[
            CategoryNode(
                name="Wedding",
                url="https://www.kapruka.com/online/flowers/wedding",
                children=[],
            ),
        ],
    ),
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


class _HybridRagMockStore:
    def __init__(self) -> None:
        self.nodes: dict[str, set[str]] = defaultdict(set)
        self.node_properties: dict[str, dict[str, Any]] = {}
        self.relationships: list[dict[str, Any]] = []
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
            self.vector_indexes[name] = {"name": name}
            return []

        if cypher.startswith("CALL db.index.vector.queryNodes"):
            query_embedding = parameters["query_embedding"]
            top_k = int(parameters["top_k"])
            hits: list[dict[str, Any]] = []
            for node_id in self.nodes.get(LABEL_CATEGORY, set()):
                props = self.node_properties.get(node_id, {})
                embedding = props.get("embedding")
                if embedding is None:
                    continue
                score = _cosine_similarity(query_embedding, embedding)
                hits.append({"id": node_id, "score": score})
            hits.sort(key=lambda row: row["score"], reverse=True)
            return hits[:top_k]

        if "WHERE c.id IN $category_ids" in cypher and "display_name" in cypher:
            rows = []
            for category_id in parameters.get("category_ids", []):
                props = self.node_properties.get(category_id, {})
                if props.get("_label") != LABEL_CATEGORY:
                    continue
                rows.append(
                    {
                        "id": category_id,
                        "display_name": props.get("display_name", category_id),
                    }
                )
            return rows

        if cypher.startswith("MATCH (seed:Category)") and "rels*1" in cypher:
            return self._traverse(parameters)

        return []

    def _traverse(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        category_ids = list(parameters.get("category_ids", []))
        max_hops = int(parameters.get("max_hops", 2))
        rel_types = set(parameters.get("rel_types", []))
        node_labels = set(parameters.get("node_labels", []))
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for seed_id in category_ids:
            if seed_id not in self.node_properties:
                continue
            frontier: list[tuple[str, int, list[dict[str, Any]]]] = [(seed_id, 0, [])]
            visited: set[str] = {seed_id}

            while frontier:
                current_id, hop, path_rels = frontier.pop(0)
                if hop >= max_hops:
                    continue
                for rel in self.relationships:
                    if rel["type"] not in rel_types:
                        continue
                    next_id: str | None = None
                    if rel["from"] == current_id:
                        next_id = rel["to"]
                    elif rel["to"] == current_id:
                        next_id = rel["from"]
                    if next_id is None or next_id in visited:
                        continue
                    visited.add(next_id)
                    props = self.node_properties.get(next_id, {})
                    label = props.get("_label", "")
                    if label not in node_labels:
                        continue
                    new_hop = hop + 1
                    new_path = [*path_rels, rel]
                    key = (seed_id, next_id, str(new_hop))
                    if key in seen:
                        continue
                    seen.add(key)
                    weight = 1.0
                    for path_rel in new_path:
                        weight *= float(path_rel.get("weight", 1.0))
                    results.append(
                        {
                            "seed_id": seed_id,
                            "id": next_id,
                            "label": label,
                            "display_name": props.get("display_name", next_id),
                            "hop": new_hop,
                            "relationship_type": new_path[-1]["type"],
                            "weight": weight,
                        }
                    )
                    frontier.append((next_id, new_hop, new_path))
        return results

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
        weight = float(row.get("weight", 1.0))
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
        self.relationships.append(
            {"from": occ_id, "to": cat_id, "type": REL_OCCASION_TO_CATEGORY, "weight": weight}
        )
        self.relationships.append(
            {
                "from": cat_id,
                "to": pt_id,
                "type": REL_CATEGORY_TO_PRODUCT_TYPE,
                "weight": weight,
            }
        )

    def _store_node(self, label: str, node_id: str, props: dict[str, Any]) -> None:
        self.nodes[label].add(node_id)
        existing = self.node_properties.setdefault(node_id, {})
        existing.update(props)
        existing["_label"] = label
        existing["id"] = node_id


class _HybridRagMockSession:
    def __init__(self, store: _HybridRagMockStore) -> None:
        self._store = store

    async def run(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _MockAsyncResult:
        del kwargs
        return _MockAsyncResult(self._store.respond(cypher.strip(), parameters or {}))

    async def __aenter__(self) -> _HybridRagMockSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _HybridRagMockDriver:
    def __init__(self, store: _HybridRagMockStore) -> None:
        self._store = store

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncIterator[_HybridRagMockSession]:
        del kwargs
        yield _HybridRagMockSession(self._store)

    async def close(self) -> None:
        return None


def _client_with_store(store: _HybridRagMockStore) -> Neo4jClient:
    return Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=_HybridRagMockDriver(store),  # type: ignore[arg-type]
    )


@pytest.fixture
async def hybrid_rag_client() -> Neo4jClient:
    store = _HybridRagMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _WEDDING_FLOWERS_TREE)
    await embed_ontology_nodes(client, embed_fn=_semantic_fake_embed)
    await create_category_vector_index(client)
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_retrieve_hybrid_context_wedding_flowers_category_filter(
    hybrid_rag_client: Neo4jClient,
) -> None:
    """'wedding flowers' should yield Flowers category hint for MCP search filters."""
    state: AgentState = {
        "messages": [HumanMessage(content="wedding flowers")],
        "intent": "discovery",
        "session_id": "sess-hybrid-rag-001",
    }

    result = await retrieve_hybrid_context(
        state,
        neo4j_client=hybrid_rag_client,
        embed_fn=_semantic_fake_embed,
    )

    hybrid_context = result["hybrid_context"]
    assert hybrid_context["hints"]["category"] == "Flowers"
    assert hybrid_context["hints"]["occasion"] == "Wedding"
    assert hybrid_context["vector_hits"][0]["display_name"] == "Flowers"
    assert any(occasion["display_name"] == "Wedding" for occasion in hybrid_context["occasions"])
