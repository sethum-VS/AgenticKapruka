"""Optional GPU-accelerated Louvain community detection via cuGraph.

Production Cloud Run deploys use the CPU NetworkX path in ``networkx_worker``.
This module imports cuGraph only when CUDA is visible and the package is installed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import logging
import os
from typing import TYPE_CHECKING, Any, Final, Literal

from lib.analytics.networkx_worker import CoPurchaseEdge

if TYPE_CHECKING:
    import cugraph as cugraph_module

logger = logging.getLogger(__name__)

AnalyticsBackend = Literal["cugraph", "networkx"]

_CUGRAPH_MODULE: cugraph_module | None = None
_CUGRAPH_IMPORT_ATTEMPTED: bool = False
_CUDF_MODULE: Any = None

_LOUVAIN_SEED: Final = 42


def is_cuda_available() -> bool:
    """Return True when an NVIDIA driver / CUDA runtime is visible."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    lib_path = ctypes.util.find_library("cuda")
    if lib_path is None:
        return False
    try:
        ctypes.CDLL(lib_path)
    except OSError:
        return False
    return True


def cugraph_available() -> bool:
    """Return True when cuGraph can be imported and CUDA is usable."""
    if not is_cuda_available():
        return False
    return _load_cugraph() is not None


def preferred_backend() -> AnalyticsBackend:
    """Return the analytics backend this process should use."""
    if cugraph_available():
        return "cugraph"
    return "networkx"


def _load_cugraph() -> cugraph_module | None:
    global _CUGRAPH_MODULE, _CUGRAPH_IMPORT_ATTEMPTED, _CUDF_MODULE
    if _CUGRAPH_IMPORT_ATTEMPTED:
        return _CUGRAPH_MODULE
    _CUGRAPH_IMPORT_ATTEMPTED = True
    try:
        _CUDF_MODULE = importlib.import_module("cudf")
        _CUGRAPH_MODULE = importlib.import_module("cugraph")
    except ImportError:
        logger.debug("cuGraph not installed; using NetworkX backend")
        return None
    return _CUGRAPH_MODULE


def detect_louvain_communities_gpu(
    edges: list[CoPurchaseEdge],
) -> list[frozenset[str]] | None:
    """Run Louvain on GPU; return None when cuGraph/CUDA is unavailable."""
    cugraph = _load_cugraph()
    if cugraph is None or _CUDF_MODULE is None:
        return None
    if not edges:
        return []

    node_ids = sorted({edge.source_id for edge in edges} | {edge.target_id for edge in edges})
    index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}

    frame = _CUDF_MODULE.DataFrame(
        {
            "src": [index_by_id[edge.source_id] for edge in edges],
            "dst": [index_by_id[edge.target_id] for edge in edges],
            "weight": [edge.weight for edge in edges],
        }
    )
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(frame, source="src", destination="dst", edge_attr="weight")

    partition, _modularity = cugraph.louvain(graph, random_state=_LOUVAIN_SEED)
    vertex_indices = partition.index.to_arrow().to_pylist()
    community_ids = partition.to_arrow().to_pylist()
    communities_by_id: dict[int, set[str]] = {}
    for vertex_index, community_id in zip(vertex_indices, community_ids, strict=True):
        communities_by_id.setdefault(int(community_id), set()).add(node_ids[int(vertex_index)])

    return [frozenset(members) for members in communities_by_id.values()]
