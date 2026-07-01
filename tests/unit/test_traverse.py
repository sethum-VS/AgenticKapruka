"""Unit tests for Neo4j ontology traversal."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import (
    LABEL_CATEGORY,
    LABEL_OCCASION,
    LABEL_PRODUCT_TYPE,
    REL_CATEGORY_TO_PRODUCT_TYPE,
    REL_OCCASION_TO_CATEGORY,
)
from lib.neo4j.traverse import _traverse_from_categories_cypher, traverse_from_categories


class _MockAsyncResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def data(self) -> list[dict[str, Any]]:
        return self._rows


class _TraverseMockStore:
    def __init__(self) -> None:
        self.node_properties: dict[str, dict[str, Any]] = {}
        self.relationships: list[dict[str, Any]] = []

    def respond(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        if cypher.startswith("MATCH (seed:Category)") and "rels*1" in cypher:
            return self._traverse(parameters, cypher=cypher)
        return []

    def _traverse(self, parameters: dict[str, Any], *, cypher: str = "") -> list[dict[str, Any]]:
        category_ids = list(parameters.get("category_ids", []))
        match = re.search(r"rels\*1\.\.(\d+)", cypher)
        max_hops = int(match.group(1)) if match else 2
        rel_types = set(parameters.get("rel_types", []))
        node_labels = set(parameters.get("node_labels", []))
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for seed_id in category_ids:
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

    def seed_fixture(self) -> None:
        """Occasion-linked categories with product type; cakes reachable in 2 hops."""
        self.node_properties = {
            "category:flowers": {
                "id": "category:flowers",
                "_label": LABEL_CATEGORY,
                "display_name": "Flowers",
            },
            "category:cakes": {
                "id": "category:cakes",
                "_label": LABEL_CATEGORY,
                "display_name": "Cakes",
            },
            "occasion:wedding": {
                "id": "occasion:wedding",
                "_label": LABEL_OCCASION,
                "display_name": "Wedding",
            },
            "product_type:flowers-wedding": {
                "id": "product_type:flowers-wedding",
                "_label": LABEL_PRODUCT_TYPE,
                "display_name": "Wedding Bouquets",
            },
        }
        self.relationships = [
            {
                "from": "occasion:wedding",
                "to": "category:flowers",
                "type": REL_OCCASION_TO_CATEGORY,
                "weight": 1.0,
            },
            {
                "from": "occasion:wedding",
                "to": "category:cakes",
                "type": REL_OCCASION_TO_CATEGORY,
                "weight": 0.8,
            },
            {
                "from": "category:flowers",
                "to": "product_type:flowers-wedding",
                "type": REL_CATEGORY_TO_PRODUCT_TYPE,
                "weight": 0.9,
            },
        ]


class _TraverseMockSession:
    def __init__(self, store: _TraverseMockStore) -> None:
        self._store = store

    async def run(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _MockAsyncResult:
        del kwargs
        return _MockAsyncResult(self._store.respond(cypher.strip(), parameters or {}))

    async def __aenter__(self) -> _TraverseMockSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _TraverseMockDriver:
    def __init__(self, store: _TraverseMockStore) -> None:
        self._store = store

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncIterator[_TraverseMockSession]:
        del kwargs
        yield _TraverseMockSession(self._store)

    async def close(self) -> None:
        return None


def _client_with_store(store: _TraverseMockStore) -> Neo4jClient:
    return Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=_TraverseMockDriver(store),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_traverse_from_categories_reaches_occasion_and_product_type() -> None:
    store = _TraverseMockStore()
    store.seed_fixture()
    client = _client_with_store(store)

    result = await traverse_from_categories(client, ["category:flowers"], max_hops=2)

    labels = {node.label for node in result.nodes}
    assert LABEL_OCCASION in labels
    assert LABEL_PRODUCT_TYPE in labels
    assert LABEL_CATEGORY in labels

    occasion = next(node for node in result.occasions if node.display_name == "Wedding")
    assert occasion.hop == 1
    assert occasion.relationship_type == REL_OCCASION_TO_CATEGORY
    assert occasion.weight == 1.0
    assert occasion.seed_id == "category:flowers"

    product_type = result.product_types[0]
    assert product_type.hop == 1
    assert product_type.relationship_type == REL_CATEGORY_TO_PRODUCT_TYPE
    assert product_type.weight == pytest.approx(0.9)

    cakes = next(node for node in result.categories if node.id == "category:cakes")
    assert cakes.hop == 2
    assert cakes.relationship_type == REL_OCCASION_TO_CATEGORY
    assert cakes.weight == pytest.approx(0.8)
    await client.close()


@pytest.mark.asyncio
async def test_traverse_from_categories_empty_seeds_returns_empty() -> None:
    store = _TraverseMockStore()
    store.seed_fixture()
    client = _client_with_store(store)

    result = await traverse_from_categories(client, [], max_hops=2)

    assert result.nodes == ()
    await client.close()


@pytest.mark.asyncio
async def test_traverse_cypher_uses_literal_hop_depth() -> None:
    cypher = _traverse_from_categories_cypher(2)
    assert "$max_hops" not in cypher
    assert "rels*1..2" in cypher

    one_hop = _traverse_from_categories_cypher(1)
    assert "rels*1..1" in one_hop


@pytest.mark.asyncio
async def test_traverse_from_categories_rejects_invalid_max_hops() -> None:
    store = _TraverseMockStore()
    client = _client_with_store(store)

    with pytest.raises(ValueError, match="max_hops"):
        await traverse_from_categories(client, ["category:flowers"], max_hops=0)

    with pytest.raises(ValueError, match="max_hops"):
        await traverse_from_categories(client, ["category:flowers"], max_hops=3)

    await client.close()
