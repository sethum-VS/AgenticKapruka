"""Neo4j client utilities."""

from lib.neo4j.client import Neo4jClient
from lib.neo4j.embed_ontology import (
    embed_ontology_nodes,
    has_category_embeddings,
)
from lib.neo4j.hybrid_context import (
    VECTOR_CONFIDENCE_THRESHOLD,
    build_discovery_search_args,
    build_graph_hybrid_context,
    fetch_category_display_names,
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
from lib.neo4j.vector_search import (
    VectorSearchHit,
    create_category_vector_index,
    has_category_vector_index,
    vector_search,
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
    "VECTOR_CONFIDENCE_THRESHOLD",
    "VectorSearchHit",
    "build_discovery_search_args",
    "build_graph_hybrid_context",
    "create_category_vector_index",
    "embed_ontology_nodes",
    "fetch_category_display_names",
    "has_category_embeddings",
    "has_category_vector_index",
    "ingest_category_tree",
    "TraversalNode",
    "TraversalResult",
    "traverse_from_categories",
    "vector_search",
    "verify_ontology_schema",
]
