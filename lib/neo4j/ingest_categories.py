"""Map Kapruka category tree to ontology triplets and MERGE into Neo4j."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlparse

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
SET c.slug = row.category_slug,
    c.display_name = row.category_display_name,
    c.description = row.category_description,
    c.kapruka_id = row.category_kapruka_id
MERGE (o:{LABEL_OCCASION} {{id: row.occasion_id}})
SET o.slug = row.occasion_slug,
    o.display_name = row.occasion_display_name,
    o.description = row.occasion_description,
    o.kapruka_id = row.occasion_kapruka_id
MERGE (p:{LABEL_PRODUCT_TYPE} {{id: row.product_type_id}})
SET p.slug = row.product_type_slug,
    p.display_name = row.product_type_display_name,
    p.description = row.product_type_description,
    p.kapruka_id = row.product_type_kapruka_id
MERGE (o)-[r1:{REL_OCCASION_TO_CATEGORY}]->(c)
SET r1.weight = row.weight
MERGE (c)-[r2:{REL_CATEGORY_TO_PRODUCT_TYPE}]->(p)
SET r2.weight = row.weight
""".strip()

_MERGE_CATEGORY_ONLY_BATCH_CYPHER = f"""
UNWIND $batch AS row
MERGE (c:{LABEL_CATEGORY} {{id: row.category_id}})
SET c.slug = row.slug,
    c.display_name = row.display_name,
    c.description = row.description,
    c.kapruka_id = row.kapruka_id
""".strip()

_FETCH_NODE_PROPERTIES_CYPHER = """
MATCH (n)
WHERE $label IN labels(n) AND n.id = $node_id
RETURN n.id AS id,
       n.slug AS slug,
       n.display_name AS display_name,
       n.description AS description,
       n.kapruka_id AS kapruka_id
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


@dataclass(frozen=True, slots=True)
class NodeEnrichment:
    """Enrichment properties derived from Kapruka category metadata."""

    slug: str
    display_name: str
    description: str
    kapruka_id: str


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


def kapruka_id_from_url(url: str) -> str:
    """Extract the trailing Kapruka path segment from a category browse URL."""
    path = urlparse(url).path.rstrip("/")
    segment = path.rsplit("/", maxsplit=1)[-1] if path else ""
    return segment or "unknown"


def build_node_description(
    label: str,
    display_name: str,
    *,
    category_name: str | None = None,
    occasion_name: str | None = None,
) -> str:
    """Build a short description for embedding and display from category context."""
    name = display_name.strip()
    if label == LABEL_CATEGORY:
        return f"Kapruka {name} — browse gifts and products in this department."
    if label == LABEL_OCCASION:
        dept = category_name or "Kapruka"
        return f"{name} — shop {name.lower()} gifts and products in {dept}."
    if label == LABEL_PRODUCT_TYPE:
        occ = occasion_name or "shopping"
        dept = category_name or "Kapruka"
        return f"{name} — {name.lower()} products for {occ} in {dept}."
    return name


def _enrichment_from_node(
    node: CategoryNode,
    label: str,
    *,
    category_name: str | None = None,
    occasion_name: str | None = None,
) -> NodeEnrichment:
    display_name = node.name.strip()
    return NodeEnrichment(
        slug=slugify_name(display_name),
        display_name=display_name,
        description=build_node_description(
            label,
            display_name,
            category_name=category_name,
            occasion_name=occasion_name,
        ),
        kapruka_id=kapruka_id_from_url(node.url),
    )


def _synthesized_product_type_enrichment(
    root: CategoryNode,
    child: CategoryNode,
) -> NodeEnrichment:
    """Enrichment for a two-level branch where ProductType is synthesized."""
    display_name = child.name.strip()
    return NodeEnrichment(
        slug=slugify_name(display_name),
        display_name=display_name,
        description=build_node_description(
            LABEL_PRODUCT_TYPE,
            display_name,
            category_name=root.name.strip(),
            occasion_name=display_name,
        ),
        kapruka_id=kapruka_id_from_url(child.url),
    )


def collect_node_enrichments(
    categories: list[CategoryNode],
) -> dict[str, NodeEnrichment]:
    """Map ontology node id → enrichment properties for each node type."""
    enrichments: dict[str, NodeEnrichment] = {}

    for root in categories:
        cat_id = category_node_id(root.name)
        enrichments[cat_id] = _enrichment_from_node(root, LABEL_CATEGORY)

        for child in root.children:
            occ_id = occasion_node_id(child.name)
            enrichments[occ_id] = _enrichment_from_node(
                child,
                LABEL_OCCASION,
                category_name=root.name.strip(),
            )

            if child.children:
                for grandchild in child.children:
                    pt_id = f"product_type:{slugify_name(grandchild.name)}"
                    enrichments[pt_id] = _enrichment_from_node(
                        grandchild,
                        LABEL_PRODUCT_TYPE,
                        category_name=root.name.strip(),
                        occasion_name=child.name.strip(),
                    )
            else:
                pt_id = product_type_node_id(root.name, child.name)
                enrichments[pt_id] = _synthesized_product_type_enrichment(root, child)

    return enrichments


def _triplet_row(
    triplet: OntologyTriplet,
    enrichments: dict[str, NodeEnrichment],
) -> dict[str, object]:
    category = enrichments[triplet.category_id]
    occasion = enrichments[triplet.occasion_id]
    product_type = enrichments[triplet.product_type_id]
    return {
        "category_id": triplet.category_id,
        "occasion_id": triplet.occasion_id,
        "product_type_id": triplet.product_type_id,
        "weight": triplet.weight,
        "category_slug": category.slug,
        "category_display_name": category.display_name,
        "category_description": category.description,
        "category_kapruka_id": category.kapruka_id,
        "occasion_slug": occasion.slug,
        "occasion_display_name": occasion.display_name,
        "occasion_description": occasion.description,
        "occasion_kapruka_id": occasion.kapruka_id,
        "product_type_slug": product_type.slug,
        "product_type_display_name": product_type.display_name,
        "product_type_description": product_type.description,
        "product_type_kapruka_id": product_type.kapruka_id,
    }


def _category_row(category_id: str, enrichments: dict[str, NodeEnrichment]) -> dict[str, object]:
    props = enrichments[category_id]
    return {
        "category_id": category_id,
        "slug": props.slug,
        "display_name": props.display_name,
        "description": props.description,
        "kapruka_id": props.kapruka_id,
    }


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
    enrichments: dict[str, NodeEnrichment],
    *,
    batch_size: int = _MERGE_BATCH_SIZE,
) -> int:
    """MERGE Category nodes that have no subcategory triplets."""
    batches = _chunked(
        [_category_row(category_id, enrichments) for category_id in category_ids],
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
    enrichments: dict[str, NodeEnrichment],
    *,
    batch_size: int = _MERGE_BATCH_SIZE,
) -> int:
    """MERGE Occasion, Category, ProductType nodes and relationship edges in batches."""
    rows = [_triplet_row(triplet, enrichments) for triplet in triplets]
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
    enrichments = collect_node_enrichments(categories)
    categories_merged = await merge_category_nodes(client, category_only, enrichments)
    triplets_merged = await merge_ontology_triplets(client, triplets, enrichments)
    return IngestStats(
        categories_merged=categories_merged,
        triplets_merged=triplets_merged,
    )


async def fetch_ontology_node_properties(
    client: Neo4jClient,
    *,
    label: str,
    node_id: str,
) -> dict[str, Any] | None:
    """Return enrichment properties for a single ontology node, or None if missing."""
    rows = await client.execute(
        _FETCH_NODE_PROPERTIES_CYPHER,
        {"label": label, "node_id": node_id},
    )
    return rows[0] if rows else None


async def count_ontology_nodes_by_label(client: Neo4jClient) -> dict[str, int]:
    """Return node counts keyed by ontology label (Occasion, Category, ProductType)."""
    rows = await client.execute(
        _COUNT_ONTOLOGY_NODES_CYPHER,
        {"ontology_labels": list(_ONTOLOGY_LABELS)},
    )
    return {row["label"]: int(row["count"]) for row in rows}
