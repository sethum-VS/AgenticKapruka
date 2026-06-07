"""Neo4j client utilities."""

from lib.neo4j.client import Neo4jClient
from lib.neo4j.ingest_categories import (
    INGEST_CATEGORY_DEPTH,
    build_triplets_from_categories,
    count_ontology_nodes_by_label,
    ingest_category_tree,
)
from lib.neo4j.ontology import (
    LABEL_CATEGORY,
    LABEL_OCCASION,
    LABEL_PRODUCT_TYPE,
    REL_CATEGORY_TO_PRODUCT_TYPE,
    REL_OCCASION_TO_CATEGORY,
    apply_ontology_schema,
    verify_ontology_schema,
)

__all__ = [
    "Neo4jClient",
    "INGEST_CATEGORY_DEPTH",
    "LABEL_CATEGORY",
    "LABEL_OCCASION",
    "LABEL_PRODUCT_TYPE",
    "REL_CATEGORY_TO_PRODUCT_TYPE",
    "REL_OCCASION_TO_CATEGORY",
    "apply_ontology_schema",
    "build_triplets_from_categories",
    "count_ontology_nodes_by_label",
    "ingest_category_tree",
    "verify_ontology_schema",
]
