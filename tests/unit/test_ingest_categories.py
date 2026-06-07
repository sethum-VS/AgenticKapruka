"""Tests for Kapruka category → Neo4j ontology ingestion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from lib.kapruka.types import CategoryNode
from lib.neo4j.client import Neo4jClient
from lib.neo4j.ingest_categories import (
    INGEST_CATEGORY_DEPTH,
    build_triplets_from_categories,
    collect_node_enrichments,
    count_ontology_nodes_by_label,
    fetch_ontology_node_properties,
    ingest_category_tree,
    kapruka_id_from_url,
    merge_category_nodes,
    merge_ontology_triplets,
    occasion_node_id,
    product_type_node_id,
    slugify_name,
)
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
    CategoryNode(
        name="Flowers",
        url="https://www.kapruka.com/online/flowers",
        children=[],
    ),
]


class _MockAsyncResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def data(self) -> list[dict[str, Any]]:
        return self._rows


class _IngestMockStore:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.nodes: dict[str, set[str]] = defaultdict(set)
        self.node_properties: dict[str, dict[str, Any]] = {}
        self.relationships: list[tuple[str, str, str]] = []

    def respond(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        if cypher.startswith("UNWIND $batch"):
            for row in parameters.get("batch", []):
                if "occasion_id" in row:
                    self._merge_triplet(row)
                elif "category_id" in row:
                    self._merge_category(row)
            return []

        if "RETURN labels(n)[0] AS label" in cypher:
            return [
                {"label": label, "count": len(ids)}
                for label, ids in sorted(self.nodes.items())
                if ids
            ]

        if "RETURN n.id AS id" in cypher:
            node_id = parameters["node_id"]
            props = self.node_properties.get(node_id)
            return [props] if props else []

        return []

    def _store_node(self, label: str, node_id: str, props: dict[str, Any]) -> None:
        self.nodes[label].add(node_id)
        self.node_properties[node_id] = {"id": node_id, **props}

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
        self.relationships.append((occ_id, REL_OCCASION_TO_CATEGORY, cat_id))
        self.relationships.append((cat_id, REL_CATEGORY_TO_PRODUCT_TYPE, pt_id))


class _IngestMockSession:
    def __init__(self, store: _IngestMockStore) -> None:
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

    async def __aenter__(self) -> _IngestMockSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _IngestMockDriver:
    def __init__(self, store: _IngestMockStore) -> None:
        self._store = store

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncIterator[_IngestMockSession]:
        del kwargs
        yield _IngestMockSession(self._store)

    async def close(self) -> None:
        return None


def _client_with_store(store: _IngestMockStore) -> Neo4jClient:
    return Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=_IngestMockDriver(store),  # type: ignore[arg-type]
    )


def test_slugify_name_normalizes_display_names() -> None:
    assert slugify_name("Birthday Cakes") == "birthday-cakes"
    assert slugify_name("  Tea & Coffee  ") == "tea-coffee"


def test_kapruka_id_from_url_extracts_trailing_segment() -> None:
    assert kapruka_id_from_url("https://www.kapruka.com/online/cakes") == "cakes"
    assert (
        kapruka_id_from_url("https://www.kapruka.com/online/cakes/price/kapruka_cakes")
        == "kapruka_cakes"
    )


def test_collect_node_enrichments_maps_birthday_occasion() -> None:
    enrichments = collect_node_enrichments(_SAMPLE_TREE)
    birthday_id = occasion_node_id("Birthday")
    birthday = enrichments[birthday_id]

    assert birthday.slug == "birthday"
    assert birthday.display_name == "Birthday"
    assert birthday.kapruka_id == "birthday"
    assert "Birthday" in birthday.description


def test_build_triplets_maps_two_level_tree() -> None:
    category_only, triplets = build_triplets_from_categories(_SAMPLE_TREE)

    assert category_only == ["category:flowers"]
    assert len(triplets) == 2
    assert triplets[0].category_id == "category:cakes"
    assert triplets[0].occasion_id == occasion_node_id("Birthday")
    assert triplets[0].product_type_id == product_type_node_id("Cakes", "Birthday")
    assert triplets[1].occasion_id == occasion_node_id("Wedding")


def test_build_triplets_maps_three_levels_when_grandchildren_present() -> None:
    tree = [
        CategoryNode(
            name="Gifts",
            url="https://example.com/gifts",
            children=[
                CategoryNode(
                    name="Birthday",
                    url="https://example.com/gifts/birthday",
                    children=[
                        CategoryNode(
                            name="Hampers",
                            url="https://example.com/gifts/birthday/hampers",
                            children=[],
                        ),
                    ],
                ),
            ],
        ),
    ]
    category_only, triplets = build_triplets_from_categories(tree)

    assert category_only == []
    assert len(triplets) == 1
    assert triplets[0].occasion_id == "occasion:birthday"
    assert triplets[0].product_type_id == "product_type:hampers"


def test_ingest_category_depth_matches_mcp_limit() -> None:
    assert INGEST_CATEGORY_DEPTH == 2


async def test_merge_ontology_triplets_executes_unwind_batches() -> None:
    store = _IngestMockStore()
    client = _client_with_store(store)
    _, triplets = build_triplets_from_categories(_SAMPLE_TREE)
    enrichments = collect_node_enrichments(_SAMPLE_TREE)

    merged = await merge_ontology_triplets(client, triplets, enrichments)

    assert merged == 2
    unwind_calls = [cypher for cypher, _ in store.executed if cypher.startswith("UNWIND")]
    assert len(unwind_calls) == 1
    assert LABEL_OCCASION in unwind_calls[0]
    assert REL_OCCASION_TO_CATEGORY in unwind_calls[0]
    assert len(store.relationships) == 4

    await client.close()


async def test_merge_category_nodes_merges_leaf_categories() -> None:
    store = _IngestMockStore()
    client = _client_with_store(store)
    enrichments = collect_node_enrichments(_SAMPLE_TREE)

    merged = await merge_category_nodes(client, ["category:flowers"], enrichments)

    assert merged == 1
    assert "category:flowers" in store.nodes[LABEL_CATEGORY]
    await client.close()


async def test_ingest_category_tree_merges_triplets_and_standalone_categories() -> None:
    store = _IngestMockStore()
    client = _client_with_store(store)

    stats = await ingest_category_tree(client, _SAMPLE_TREE)

    assert stats.triplets_merged == 2
    assert stats.categories_merged == 1
    assert len(store.nodes[LABEL_CATEGORY]) == 2
    assert len(store.nodes[LABEL_OCCASION]) == 2
    assert len(store.nodes[LABEL_PRODUCT_TYPE]) == 2

    counts = await count_ontology_nodes_by_label(client)
    assert counts[LABEL_CATEGORY] == 2
    assert counts[LABEL_OCCASION] == 2
    assert counts[LABEL_PRODUCT_TYPE] == 2

    await client.close()


async def test_count_ontology_nodes_by_label_returns_label_counts() -> None:
    store = _IngestMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)

    counts = await count_ontology_nodes_by_label(client)

    assert sum(counts.values()) == 6
    await client.close()


async def test_birthday_occasion_node_has_display_name_and_slug() -> None:
    """PRD-043: enriched Occasion nodes carry slug and display_name from Kapruka metadata."""
    store = _IngestMockStore()
    client = _client_with_store(store)
    await ingest_category_tree(client, _SAMPLE_TREE)

    birthday_id = occasion_node_id("Birthday")
    props = await fetch_ontology_node_properties(
        client,
        label=LABEL_OCCASION,
        node_id=birthday_id,
    )

    assert props is not None
    assert props["slug"] == "birthday"
    assert props["display_name"] == "Birthday"
    assert props["kapruka_id"] == "birthday"
    assert props["description"]

    await client.close()
