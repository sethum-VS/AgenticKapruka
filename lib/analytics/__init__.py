"""Graph analytics workers for recommendation enrichment."""

from lib.analytics.cugraph_optional import (
    cugraph_available,
    detect_louvain_communities_gpu,
    is_cuda_available,
    preferred_backend,
)
from lib.analytics.networkx_worker import (
    DEFAULT_INTERVAL_SECONDS,
    CommunityDetectionResult,
    CoPurchaseEdge,
    NetworkXCommunityWorker,
    build_networkx_graph,
    build_recommendation_rows,
    detect_louvain_communities,
    fetch_co_purchase_edges,
    persist_recommends,
    run_community_detection,
    synthesize_category_proximity_edges,
)

__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "CoPurchaseEdge",
    "CommunityDetectionResult",
    "NetworkXCommunityWorker",
    "build_networkx_graph",
    "build_recommendation_rows",
    "cugraph_available",
    "detect_louvain_communities",
    "detect_louvain_communities_gpu",
    "fetch_co_purchase_edges",
    "is_cuda_available",
    "persist_recommends",
    "preferred_backend",
    "run_community_detection",
    "synthesize_category_proximity_edges",
]
