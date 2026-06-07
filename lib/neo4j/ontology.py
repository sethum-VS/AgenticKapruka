"""Neo4j GraphRAG ontology: node labels, relationships, constraints, and property schemas."""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, Field

from lib.neo4j.client import Neo4jClient

# --- Node labels ---

LABEL_OCCASION: Final = "Occasion"
LABEL_CATEGORY: Final = "Category"
LABEL_PRODUCT_TYPE: Final = "ProductType"

NODE_LABELS: Final[tuple[str, ...]] = (
    LABEL_OCCASION,
    LABEL_CATEGORY,
    LABEL_PRODUCT_TYPE,
)

# --- Relationship types ---

REL_OCCASION_TO_CATEGORY: Final = "OCCASION_TO_CATEGORY"
REL_CATEGORY_TO_PRODUCT_TYPE: Final = "CATEGORY_TO_PRODUCT_TYPE"

RELATIONSHIP_TYPES: Final[tuple[str, ...]] = (
    REL_OCCASION_TO_CATEGORY,
    REL_CATEGORY_TO_PRODUCT_TYPE,
)

# --- Constraint names (unique `id` per label) ---

CONSTRAINT_OCCASION_ID: Final = "occasion_id_unique"
CONSTRAINT_CATEGORY_ID: Final = "category_id_unique"
CONSTRAINT_PRODUCT_TYPE_ID: Final = "product_type_id_unique"

CONSTRAINT_NAMES: Final[tuple[str, ...]] = (
    CONSTRAINT_OCCASION_ID,
    CONSTRAINT_CATEGORY_ID,
    CONSTRAINT_PRODUCT_TYPE_ID,
)

CONSTRAINT_STATEMENTS: Final[tuple[str, ...]] = (
    f"CREATE CONSTRAINT {CONSTRAINT_OCCASION_ID} IF NOT EXISTS "
    f"FOR (n:{LABEL_OCCASION}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT {CONSTRAINT_CATEGORY_ID} IF NOT EXISTS "
    f"FOR (n:{LABEL_CATEGORY}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT {CONSTRAINT_PRODUCT_TYPE_ID} IF NOT EXISTS "
    f"FOR (n:{LABEL_PRODUCT_TYPE}) REQUIRE n.id IS UNIQUE",
)

_SHOW_CONSTRAINTS_CYPHER = """
SHOW CONSTRAINTS
YIELD name, type, labelsOrTypes, properties
WHERE name IN $names
RETURN name, type, labelsOrTypes, properties
""".strip()


class OntologyNodeProperties(BaseModel):
    """Shared property schema for Occasion, Category, and ProductType nodes."""

    id: str = Field(..., min_length=1, max_length=200, description="Unique ontology node id")
    slug: str | None = Field(default=None, max_length=200)
    display_name: str | None = Field(default=None, max_length=300)
    description: str | None = Field(default=None, max_length=2000)
    kapruka_id: str | None = Field(default=None, max_length=80)
    embedding: list[float] | None = Field(
        default=None,
        description="Vertex text-embedding-005 vector (768 dims); set by embed_ontology script",
    )


class OccasionToCategoryProperties(BaseModel):
    """Properties on (:Occasion)-[:OCCASION_TO_CATEGORY]->(:Category)."""

    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class CategoryToProductTypeProperties(BaseModel):
    """Properties on (:Category)-[:CATEGORY_TO_PRODUCT_TYPE]->(:ProductType)."""

    weight: float = Field(default=1.0, ge=0.0, le=1.0)


async def apply_ontology_schema(client: Neo4jClient) -> None:
    """Apply unique-id constraints for all ontology node labels."""
    for statement in CONSTRAINT_STATEMENTS:
        await client.execute(statement)


async def list_ontology_constraints(client: Neo4jClient) -> list[dict[str, Any]]:
    """Return SHOW CONSTRAINTS rows for ontology unique-id constraints."""
    return await client.execute(
        _SHOW_CONSTRAINTS_CYPHER,
        {"names": list(CONSTRAINT_NAMES)},
    )


async def verify_ontology_schema(client: Neo4jClient) -> bool:
    """Return True when all expected ontology constraints exist in Neo4j."""
    rows = await list_ontology_constraints(client)
    found = {row["name"] for row in rows}
    return set(CONSTRAINT_NAMES).issubset(found)
