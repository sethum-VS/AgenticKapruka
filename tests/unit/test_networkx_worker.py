"""Tests for NetworkX Louvain community detection worker."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import networkx as nx
import pytest

from lib.analytics.networkx_worker import (
    REL_CO_PURCHASED_WITH,
    REL_RECOMMENDS,
    CoPurchaseEdge,
    NetworkXCommunityWorker,
    build_networkx_graph,
    build_recommendation_rows,
    detect_louvain_communities,
    persist_recommends,
    run_community_detection,
    synthesize_category_proximity_edges,
)
from lib.kapruka.types import CategoryNode
from lib.neo4j.client import Neo4jClient
from lib.neo4j.ingest_categories import ingest_category_tree
from lib.neo4j.ontology import (
    LABEL_CATEGORY,
    LABEL_OCCASION,
    LABEL_PRODUCT_TYPE,
    REL_CATEGORY_TO_PRODUCT_TYPE,
    REL_OCCASION_TO_CATEGORY,
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
            CategoryNode(
                name="Wedding",
                url="https://www.kapruka.com/online/cakes/wedding",
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


class _CommunityMockStore:
    """In-memory Neo4j store for community-detection Cypher."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.nodes: dict[str, set[str]] = defaultdict(set)
        self.node_properties: dict[str, dict[str, Any]] = {}
        self.relationships: list[tuple[str, str, str, dict[str, Any]]] = []
        self.co_purchase_edges: list[CoPurchaseEdge] = []

    def respond(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        batch = parameters.get("batch", [])

        if cypher.startswith("UNWIND $batch") and REL_RECOMMENDS in cypher:
            for row in batch:
                self.relationships.append(
                    (
                        row["source_id"],
                        REL_RECOMMENDS,
                        row["target_id"],
                        {
                            "score": row["score"],
                            "community_id": row["community_id"],
                            "source": row["source"],
                        },
                    )
                )
            return []

        if cypher.startswith("UNWIND $batch") and batch and "occasion_id" in batch[0]:
            for row in batch:
                self._merge_triplet(row)
            return []

        if (
            cypher.startswith("UNWIND $batch")
            and batch
            and "category_id" in batch[0]
            and "occasion_id" not in batch[0]
        ):
            for row in batch:
                self._merge_category_only(row)
            return []

        if f":Product)-[r:{REL_CO_PURCHASED_WITH}" in cypher:
            return [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "weight": edge.weight,
                }
                for edge in self.co_purchase_edges
            ]

        if "WHERE a.id < b.id" in cypher and REL_CATEGORY_TO_PRODUCT_TYPE in cypher:
            rows: list[dict[str, Any]] = []
            category_to_types: dict[str, list[str]] = defaultdict(list)
            for source, rel, target, _props in self.relationships:
                if rel == REL_CATEGORY_TO_PRODUCT_TYPE:
                    category_to_types[source].append(target)
            for product_types in category_to_types.values():
                sorted_types = sorted(product_types)
                for left_index, left_id in enumerate(sorted_types):
                    for right_id in sorted_types[left_index + 1 :]:
                        rows.append(
                            {
                                "source_id": left_id,
                                "target_id": right_id,
                                "weight": 1.0,
                            }
                        )
            return rows

        return []

    def _store_node(self, label: str, node_id: str, props: dict[str, Any]) -> None:
        self.nodes[label].add(node_id)
        self.node_properties[node_id] = {"id": node_id, **props}

    def _merge_category_only(self, row: dict[str, Any]) -> None:
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
        self.relationships.append((occ_id, REL_OCCASION_TO_CATEGORY, cat_id, {}))
        self.relationships.append((cat_id, REL_CATEGORY_TO_PRODUCT_TYPE, pt_id, {}))

    def add_co_purchase(self, source_id: str, target_id: str, weight: float = 1.0) -> None:
        self.co_purchase_edges.append(
            CoPurchaseEdge(source_id=source_id, target_id=target_id, weight=weight)
        )
        for node_id in (source_id, target_id):
            self.nodes["Product"].add(node_id)
            self.node_properties[node_id] = {"id": node_id}


def _client_from_store(store: _CommunityMockStore) -> Neo4jClient:
    class _MockAsyncSession:
        async def run(
            self,
            cypher: str,
            parameters: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> _MockAsyncResult:
            del kwargs
            store.executed.append((cypher, parameters or {}))
            return _MockAsyncResult(store.respond(cypher, parameters or {}))

        async def __aenter__(self) -> _MockAsyncSession:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    class _MockAsyncDriver:
        @asynccontextmanager
        async def session(self, **kwargs: Any) -> AsyncIterator[_MockAsyncSession]:
            del kwargs
            yield _MockAsyncSession()

        async def close(self) -> None:
            return None

    return Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=_MockAsyncDriver(),  # type: ignore[arg-type]
    )


def test_build_networkx_graph_aggregates_weights() -> None:
    edges = [
        CoPurchaseEdge("a", "b", 1.0),
        CoPurchaseEdge("b", "c", 2.0),
    ]
    graph = build_networkx_graph(edges)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert graph["a"]["b"]["weight"] == 1.0


def test_detect_louvain_communities_splits_components() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("b", "c", weight=1.0)
    graph.add_edge("x", "y", weight=1.0)

    communities = detect_louvain_communities(graph)

    assert len(communities) == 2
    ids = {frozenset(community) for community in communities}
    assert frozenset({"a", "b", "c"}) in ids
    assert frozenset({"x", "y"}) in ids


def test_build_recommendation_rows_bidirectional_within_community() -> None:
    communities = [frozenset({"a", "b", "c"})]
    edges = [CoPurchaseEdge("a", "b", 0.9)]

    rows = build_recommendation_rows(communities, edges, edge_source="co_purchase")

    pairs = {(row["source_id"], row["target_id"]) for row in rows}
    assert ("a", "b") in pairs
    assert ("b", "a") in pairs
    assert ("a", "c") in pairs
    assert ("c", "a") in pairs
    ab_row = next(row for row in rows if row["source_id"] == "a" and row["target_id"] == "b")
    assert ab_row["score"] == 0.9
    assert ab_row["source"] == "co_purchase"


@pytest.mark.asyncio
async def test_synthesize_category_proximity_edges_from_ingested_tree() -> None:
    store = _CommunityMockStore()
    client = _client_from_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)

    edges = await synthesize_category_proximity_edges(client)

    assert len(edges) >= 1
    assert any(edge.source_id.startswith("product_type:") for edge in edges)


@pytest.mark.asyncio
async def test_run_community_detection_uses_co_purchase_when_present() -> None:
    store = _CommunityMockStore()
    store.add_co_purchase("product:a", "product:b", 2.0)
    store.add_co_purchase("product:b", "product:c", 1.5)
    client = _client_from_store(store)

    result = await run_community_detection(client)

    assert result.edge_source == "co_purchase"
    assert result.communities_found >= 1
    assert result.recommends_written > 0
    assert any(rel[1] == REL_RECOMMENDS for rel in store.relationships)


@pytest.mark.asyncio
async def test_run_community_detection_falls_back_to_category_proximity() -> None:
    store = _CommunityMockStore()
    client = _client_from_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)

    result = await run_community_detection(client)

    assert result.edge_source == "category_proximity"
    assert result.nodes_in_graph >= 2
    assert result.recommends_written > 0
    recommend = next(rel for rel in store.relationships if rel[1] == REL_RECOMMENDS)
    assert recommend[3]["source"] == "category_proximity"


@pytest.mark.asyncio
async def test_persist_recommends_batches_writes() -> None:
    store = _CommunityMockStore()
    client = _client_from_store(store)
    rows = [
        {
            "source_id": "a",
            "target_id": "b",
            "score": 0.8,
            "community_id": "0",
            "source": "co_purchase",
        }
    ]

    written = await persist_recommends(client, rows, batch_size=1)

    assert written == 1
    assert len(store.executed) == 1


@pytest.mark.asyncio
async def test_networkx_worker_run_once_delegates() -> None:
    store = _CommunityMockStore()
    store.add_co_purchase("p1", "p2")
    client = _client_from_store(store)
    worker = NetworkXCommunityWorker(client, interval_seconds=60)

    result = await worker.run_once()

    assert result.recommends_written > 0
    assert worker.is_running is False


@pytest.mark.asyncio
async def test_networkx_worker_start_and_stop() -> None:
    store = _CommunityMockStore()
    store.add_co_purchase("p1", "p2")
    client = _client_from_store(store)
    worker = NetworkXCommunityWorker(client, interval_seconds=3600)

    run_calls = 0
    original_run = worker.run_once

    async def counting_run() -> Any:
        nonlocal run_calls
        run_calls += 1
        return await original_run()

    worker.run_once = counting_run  # type: ignore[method-assign]

    await worker.start()
    assert worker.is_running is True
    await asyncio.sleep(0.05)
    await worker.stop()
    assert worker.is_running is False
    assert run_calls >= 1
