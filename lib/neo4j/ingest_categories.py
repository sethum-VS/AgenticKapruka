"""Map Kapruka category tree to ontology triplets and MERGE into Neo4j."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from lib.kapruka.types import CategoryNode
from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import (
    LABEL_CATEGORY,
    LABEL_OCCASION,
    LABEL_PRODUCT_TYPE,
    REL_CATEGORY_TO_PRODUCT_TYPE,
    REL_OCCASION_TO_CATEGORY,
)

# MCP kapruka_list_categories caps depth at 2; PRD cites depth=3 for full triplet levels.
INGEST_CATEGORY_DEPTH: Final = 2

_DEFAULT_REL_WEIGHT: Final = 1.0
_MERGE_BATCH_SIZE: Final = 50

_MERGE_TRIPLET_BATCH_CYPHER = f"""
UNWIND $batch AS row
MERGE (c:{LABEL_CATEGORY} {{id: row.category_id}})
MERGE (o:{LABEL_OCCASION} {{id: row.occasion_id}})
MERGE (p:{LABEL_PRODUCT_TYPE} {{id: row.product_type_id}})
MERGE (o)-[r1:{REL_OCCASION_TO_CATEGORY}]->(c)
SET r1.weight = row.weight
MERGE (c)-[r2:{REL_CATEGORY_TO_PRODUCT_TYPE}]->(p)
SET r2.weight = row.weight
""".strip()

_MERGE_CATEGORY_ONLY_BATCH_CYPHER = f"""
UNWIND $batch AS row
MERGE (c:{LABEL_CATEGORY} {{id: row.category_id}})
""".strip()

_COUNT_ONTOLOGY_NODES_CYPHER = """
MATCH (n)
WHERE any(l IN labels(n) WHERE l IN $ontology_labels)
RETURN labels(n)[0] AS label, count(*) AS count
ORDER BY label
""".strip()

_ONTOLOGY_LABELS: Final = (LABEL_OCCASION, LABEL_CATEGORY, LABEL_PRODUCT_TYPE)


@dataclass(frozen=True, slots=True)
class OntologyTriplet:
    """Occasion → Category → ProductType row for batched MERGE."""

    category_id: str
    occasion_id: str
    product_type_id: str
    weight: float = _DEFAULT_REL_WEIGHT


@dataclass(frozen=True, slots=True)
class IngestStats:
    """Summary counts from a category-tree ingest run."""

    categories_merged: int
    triplets_merged: int


def slugify_name(name: str) -> str:
    """Derive a stable lowercase slug from a Kapruka category display name."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-") or "unknown"


def category_node_id(name: str) -> str:
    return f"category:{slugify_name(name)}"


def occasion_node_id(name: str) -> str:
    return f"occasion:{slugify_name(name)}"


def product_type_node_id(category_name: str, subcategory_name: str) -> str:
    cat_slug = slugify_name(category_name)
    sub_slug = slugify_name(subcategory_name)
    return f"product_type:{cat_slug}-{sub_slug}"


def build_triplets_from_categories(
    categories: list[CategoryNode],
) -> tuple[list[str], list[OntologyTriplet]]:
    """Map Kapruka tree to standalone categories and Occasion-Category-ProductType triplets.

    Level 0 (root) → Category; level 1 → Occasion; level 2 → ProductType when present.
    When only two levels exist, ProductType id is synthesized from root + child names.
    """
    category_only: list[str] = []
    triplets: list[OntologyTriplet] = []

    for root in categories:
        cat_id = category_node_id(root.name)
        if not root.children:
            category_only.append(cat_id)
            continue

        for child in root.children:
            occ_id = occasion_node_id(child.name)
            if child.children:
                for grandchild in child.children:
                    pt_id = f"product_type:{slugify_name(grandchild.name)}"
                    triplets.append(
                        OntologyTriplet(
                            category_id=cat_id,
                            occasion_id=occ_id,
                            product_type_id=pt_id,
                        )
                    )
            else:
                triplets.append(
                    OntologyTriplet(
                        category_id=cat_id,
                        occasion_id=occ_id,
                        product_type_id=product_type_node_id(root.name, child.name),
                    )
                )

    return category_only, triplets


async def _merge_batches(
    client: Neo4jClient,
    *,
    cypher: str,
    batches: list[list[dict[str, object]]],
) -> int:
    merged = 0
    for batch in batches:
        if not batch:
            continue
        await client.execute(cypher, {"batch": batch})
        merged += len(batch)
    return merged


def _chunked(items: list[dict[str, object]], size: int) -> list[list[dict[str, object]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def merge_category_nodes(
    client: Neo4jClient,
    category_ids: list[str],
    *,
    batch_size: int = _MERGE_BATCH_SIZE,
) -> int:
    """MERGE Category nodes that have no subcategory triplets."""
    batches = _chunked(
        [{"category_id": category_id} for category_id in category_ids],
        batch_size,
    )
    return await _merge_batches(
        client,
        cypher=_MERGE_CATEGORY_ONLY_BATCH_CYPHER,
        batches=batches,
    )


async def merge_ontology_triplets(
    client: Neo4jClient,
    triplets: list[OntologyTriplet],
    *,
    batch_size: int = _MERGE_BATCH_SIZE,
) -> int:
    """MERGE Occasion, Category, ProductType nodes and relationship edges in batches."""
    rows = [
        {
            "category_id": triplet.category_id,
            "occasion_id": triplet.occasion_id,
            "product_type_id": triplet.product_type_id,
            "weight": triplet.weight,
        }
        for triplet in triplets
    ]
    batches = _chunked(rows, batch_size)
    return await _merge_batches(
        client,
        cypher=_MERGE_TRIPLET_BATCH_CYPHER,
        batches=batches,
    )


async def ingest_category_tree(
    client: Neo4jClient,
    categories: list[CategoryNode],
) -> IngestStats:
    """MERGE ontology nodes derived from a Kapruka category tree."""
    category_only, triplets = build_triplets_from_categories(categories)
    categories_merged = await merge_category_nodes(client, category_only)
    triplets_merged = await merge_ontology_triplets(client, triplets)
    return IngestStats(
        categories_merged=categories_merged,
        triplets_merged=triplets_merged,
    )


async def count_ontology_nodes_by_label(client: Neo4jClient) -> dict[str, int]:
    """Return node counts keyed by ontology label (Occasion, Category, ProductType)."""
    rows = await client.execute(
        _COUNT_ONTOLOGY_NODES_CYPHER,
        {"ontology_labels": list(_ONTOLOGY_LABELS)},
    )
    return {row["label"]: int(row["count"]) for row in rows}
