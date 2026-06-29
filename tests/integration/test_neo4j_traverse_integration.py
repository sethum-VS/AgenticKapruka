"""Integration tests for Neo4j traverse Cypher against a real Aura instance."""

from __future__ import annotations

import os

import pytest

from app.config import get_settings
from lib.neo4j.client import Neo4jClient
from lib.neo4j.traverse import _traverse_from_categories_cypher, traverse_from_categories

pytestmark = pytest.mark.neo4j_integration


@pytest.fixture
async def neo4j_client() -> Neo4jClient:
    if not os.environ.get("NEO4J_URI"):
        pytest.skip("NEO4J_URI not set")
    settings = get_settings()
    client = await Neo4jClient.connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    yield client
    await client.close()


def test_traverse_cypher_has_literal_hop_depth() -> None:
    """Regression: Neo4j 5.27 rejects $max_hops inside variable-length patterns."""
    cypher = _traverse_from_categories_cypher(2)
    assert "$max_hops" not in cypher
    assert "rels*1..2" in cypher


@pytest.mark.asyncio
async def test_traverse_from_categories_on_real_neo4j(neo4j_client: Neo4jClient) -> None:
    """Smoke traverse against Aura; empty seed id validates Cypher without requiring data."""
    result = await traverse_from_categories(neo4j_client, ["__health_probe__"], max_hops=2)
    assert result.nodes == ()
