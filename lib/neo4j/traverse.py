"""Multi-hop ontology traversal from seed Category nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import (
    LABEL_CATEGORY,
    LABEL_OCCASION,
    LABEL_PRODUCT_TYPE,
    REL_CATEGORY_TO_PRODUCT_TYPE,
    REL_OCCASION_TO_CATEGORY,
)

_ALLOWED_RELATIONSHIPS: Final = (
    REL_OCCASION_TO_CATEGORY,
    REL_CATEGORY_TO_PRODUCT_TYPE,
)
_ONTOLOGY_LABELS: Final = (LABEL_OCCASION, LABEL_CATEGORY, LABEL_PRODUCT_TYPE)

_TRAVERSE_FROM_CATEGORIES_CYPHER = f"""
MATCH (seed:{LABEL_CATEGORY})
WHERE seed.id IN $category_ids
MATCH path = (seed)-[rels*1..$max_hops]-(connected)
WHERE ALL(r IN rels WHERE type(r) IN $rel_types)
  AND ANY(label IN labels(connected) WHERE label IN $node_labels)
RETURN DISTINCT
  seed.id AS seed_id,
  connected.id AS id,
  labels(connected)[0] AS label,
  connected.display_name AS display_name,
  length(rels) AS hop,
  [r IN rels | type(r)][-1] AS relationship_type,
  reduce(w = 1.0, r IN rels | w * coalesce(r.weight, 1.0)) AS weight
ORDER BY hop, weight DESC, display_name
""".strip()


@dataclass(frozen=True, slots=True)
class TraversalNode:
    """Connected ontology node reached within max_hops of a seed category."""

    id: str
    label: str
    display_name: str
    hop: int
    relationship_type: str
    weight: float
    seed_id: str


@dataclass(frozen=True, slots=True)
class TraversalResult:
    """2-hop traversal results grouped by ontology label."""

    nodes: tuple[TraversalNode, ...]

    @property
    def occasions(self) -> list[TraversalNode]:
        return [node for node in self.nodes if node.label == LABEL_OCCASION]

    @property
    def categories(self) -> list[TraversalNode]:
        return [node for node in self.nodes if node.label == LABEL_CATEGORY]

    @property
    def product_types(self) -> list[TraversalNode]:
        return [node for node in self.nodes if node.label == LABEL_PRODUCT_TYPE]


async def traverse_from_categories(
    client: Neo4jClient,
    category_ids: list[str],
    *,
    max_hops: int = 2,
) -> TraversalResult:
    """Traverse up to max_hops from seed categories along ontology relationships."""
    if not category_ids:
        return TraversalResult(nodes=())

    if max_hops < 1:
        msg = "max_hops must be >= 1"
        raise ValueError(msg)

    rows = await client.execute(
        _TRAVERSE_FROM_CATEGORIES_CYPHER,
        {
            "category_ids": category_ids,
            "max_hops": max_hops,
            "rel_types": list(_ALLOWED_RELATIONSHIPS),
            "node_labels": list(_ONTOLOGY_LABELS),
        },
    )
    nodes = tuple(_row_to_traversal_node(row) for row in rows)
    return TraversalResult(nodes=nodes)


def _row_to_traversal_node(row: dict[str, Any]) -> TraversalNode:
    return TraversalNode(
        id=str(row["id"]),
        label=str(row["label"]),
        display_name=str(row.get("display_name") or row["id"]),
        hop=int(row["hop"]),
        relationship_type=str(row.get("relationship_type") or ""),
        weight=float(row.get("weight") or 1.0),
        seed_id=str(row["seed_id"]),
    )
