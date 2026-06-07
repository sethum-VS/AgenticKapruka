"""Tests for Neo4j GraphRAG ontology schema and migration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import ValidationError

from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import (
    CONSTRAINT_CATEGORY_ID,
    CONSTRAINT_NAMES,
    CONSTRAINT_OCCASION_ID,
    CONSTRAINT_PRODUCT_TYPE_ID,
    CONSTRAINT_STATEMENTS,
    LABEL_CATEGORY,
    LABEL_OCCASION,
    LABEL_PRODUCT_TYPE,
    NODE_LABELS,
    REL_CATEGORY_TO_PRODUCT_TYPE,
    REL_OCCASION_TO_CATEGORY,
    RELATIONSHIP_TYPES,
    CategoryToProductTypeProperties,
    OccasionToCategoryProperties,
    OntologyNodeProperties,
    apply_ontology_schema,
    list_ontology_constraints,
    verify_ontology_schema,
)


class _MockAsyncResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def data(self) -> list[dict[str, Any]]:
        return self._rows


class _RecordingMockSession:
    def __init__(self, store: _OntologyMockStore) -> None:
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

    async def __aenter__(self) -> _RecordingMockSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _OntologyMockStore:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.constraints: dict[str, dict[str, Any]] = {}

    def respond(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        if cypher.startswith("CREATE CONSTRAINT"):
            name = cypher.split()[2]
            label = cypher.split("FOR (n:")[1].split(")")[0]
            self.constraints[name] = {
                "name": name,
                "type": "UNIQUENESS",
                "labelsOrTypes": [label],
                "properties": ["id"],
            }
            return []

        if cypher.startswith("SHOW CONSTRAINTS"):
            names = set(parameters.get("names", []))
            return [row for row in self.constraints.values() if row["name"] in names]

        return []


class _RecordingMockDriver:
    def __init__(self, store: _OntologyMockStore) -> None:
        self._store = store

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncIterator[_RecordingMockSession]:
        del kwargs
        yield _RecordingMockSession(self._store)

    async def close(self) -> None:
        return None


def _client_with_store(store: _OntologyMockStore) -> Neo4jClient:
    return Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=_RecordingMockDriver(store),  # type: ignore[arg-type]
    )


def test_ontology_constants_cover_triplet_taxonomy() -> None:
    """Node labels and relationship types match GraphRAG triplet design."""
    assert NODE_LABELS == (LABEL_OCCASION, LABEL_CATEGORY, LABEL_PRODUCT_TYPE)
    assert RELATIONSHIP_TYPES == (REL_OCCASION_TO_CATEGORY, REL_CATEGORY_TO_PRODUCT_TYPE)
    assert len(CONSTRAINT_STATEMENTS) == len(NODE_LABELS)
    assert CONSTRAINT_NAMES == (
        CONSTRAINT_OCCASION_ID,
        CONSTRAINT_CATEGORY_ID,
        CONSTRAINT_PRODUCT_TYPE_ID,
    )


def test_constraint_statements_use_unique_id_per_label() -> None:
    """Each CREATE CONSTRAINT targets `id` uniqueness on one ontology label."""
    for statement, label in zip(CONSTRAINT_STATEMENTS, NODE_LABELS, strict=True):
        assert f"FOR (n:{label}) REQUIRE n.id IS UNIQUE" in statement
        assert "IF NOT EXISTS" in statement


def test_ontology_node_properties_schema() -> None:
    """OntologyNodeProperties validates id and optional enrichment fields."""
    node = OntologyNodeProperties(
        id="occasion:birthday",
        slug="birthday",
        display_name="Birthday",
        description="Birthday gifts and cakes",
        kapruka_id="CATSYM123",
    )
    assert node.id == "occasion:birthday"
    assert node.embedding is None

    with pytest.raises(ValidationError):
        OntologyNodeProperties(id="")


def test_relationship_property_schemas() -> None:
    """Relationship models carry traversal weight for multi-hop queries."""
    rel = OccasionToCategoryProperties(weight=0.8)
    assert rel.weight == 0.8
    cat_rel = CategoryToProductTypeProperties()
    assert cat_rel.weight == 1.0


async def test_apply_ontology_schema_runs_all_constraints() -> None:
    """apply_ontology_schema executes one CREATE CONSTRAINT per node label."""
    store = _OntologyMockStore()
    client = _client_with_store(store)

    await apply_ontology_schema(client)

    create_calls = [
        cypher for cypher, _ in store.executed if cypher.startswith("CREATE CONSTRAINT")
    ]
    assert len(create_calls) == len(CONSTRAINT_STATEMENTS)
    assert create_calls == list(CONSTRAINT_STATEMENTS)

    await client.close()


async def test_verify_ontology_schema_after_migration() -> None:
    """verify_ontology_schema returns True once all constraints are present."""
    store = _OntologyMockStore()
    client = _client_with_store(store)

    assert await verify_ontology_schema(client) is False

    await apply_ontology_schema(client)
    assert await verify_ontology_schema(client) is True

    rows = await list_ontology_constraints(client)
    assert {row["name"] for row in rows} == set(CONSTRAINT_NAMES)
    assert all(row["properties"] == ["id"] for row in rows)

    await client.close()
