"""Neo4j client utilities."""

from lib.neo4j.client import Neo4jClient
from lib.neo4j.embed_ontology import (
    embed_ontology_nodes,
    has_category_embeddings,
)
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
from lib.neo4j.traverse import (
    TraversalNode,
    TraversalResult,
    traverse_from_categories,
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
    "embed_ontology_nodes",
    "has_category_embeddings",
    "ingest_category_tree",
    "TraversalNode",
    "TraversalResult",
    "traverse_from_categories",
    "verify_ontology_schema",
]
