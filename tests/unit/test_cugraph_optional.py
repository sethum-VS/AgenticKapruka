"""Tests for optional cuGraph GPU community detection."""

from __future__ import annotations

from types import ModuleType
from typing import Any
from unittest.mock import patch

from lib.analytics.cugraph_optional import (
    cugraph_available,
    detect_louvain_communities_gpu,
    is_cuda_available,
    preferred_backend,
)
from lib.analytics.networkx_worker import CoPurchaseEdge


def test_is_cuda_available_false_without_driver() -> None:
    with patch("lib.analytics.cugraph_optional.ctypes.util.find_library", return_value=None):
        assert is_cuda_available() is False


def test_is_cuda_available_false_when_devices_hidden() -> None:
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
        assert is_cuda_available() is False


def test_cugraph_available_false_without_cuda() -> None:
    with patch("lib.analytics.cugraph_optional.is_cuda_available", return_value=False):
        assert cugraph_available() is False


def test_preferred_backend_networkx_without_cuda() -> None:
    with patch("lib.analytics.cugraph_optional.cugraph_available", return_value=False):
        assert preferred_backend() == "networkx"


def test_preferred_backend_cugraph_when_available() -> None:
    with patch("lib.analytics.cugraph_optional.cugraph_available", return_value=True):
        assert preferred_backend() == "cugraph"


def test_detect_louvain_communities_gpu_returns_none_without_cugraph() -> None:
    edges = [CoPurchaseEdge("a", "b", 1.0)]
    with patch("lib.analytics.cugraph_optional._load_cugraph", return_value=None):
        assert detect_louvain_communities_gpu(edges) is None


def test_detect_louvain_communities_gpu_empty_edges() -> None:
    fake_cugraph = ModuleType("cugraph")

    class _FakeGraph:
        def __init__(self, *, directed: bool) -> None:
            self.directed = directed

        def from_cudf_edgelist(self, *args: Any, **kwargs: Any) -> None:
            return None

    def _fake_louvain(graph: _FakeGraph, *, random_state: int) -> tuple[Any, float]:
        del graph, random_state
        raise AssertionError("louvain should not run for empty edge list")

    fake_cugraph.Graph = _FakeGraph  # type: ignore[attr-defined]
    fake_cugraph.louvain = _fake_louvain  # type: ignore[attr-defined]

    with (
        patch("lib.analytics.cugraph_optional._load_cugraph", return_value=fake_cugraph),
        patch("lib.analytics.cugraph_optional._CUDF_MODULE", object()),
    ):
        assert detect_louvain_communities_gpu([]) == []


def test_detect_louvain_communities_gpu_partitions_mocked() -> None:
    fake_cugraph = ModuleType("cugraph")

    class _FakeGraph:
        def __init__(self, *, directed: bool) -> None:
            self.directed = directed

        def from_cudf_edgelist(self, *args: Any, **kwargs: Any) -> None:
            return None

    class _FakeSeries:
        def __init__(self, values: list[int]) -> None:
            self._values = values
            self.index = _FakeIndex(list(range(len(values))))

        def to_arrow(self) -> _FakeArrow:
            return _FakeArrow(self._values)

    class _FakeIndex:
        def __init__(self, values: list[int]) -> None:
            self._values = values

        def to_arrow(self) -> _FakeArrow:
            return _FakeArrow(self._values)

    class _FakeArrow:
        def __init__(self, values: list[int]) -> None:
            self._values = values

        def to_pylist(self) -> list[int]:
            return self._values

    class _FakeDataFrame:
        def __init__(self, data: dict[str, list[Any]]) -> None:
            self.data = data

    def _fake_louvain(graph: _FakeGraph, *, random_state: int) -> tuple[_FakeSeries, float]:
        del graph, random_state
        # Five vertices (a,b,c,x,y) -> two communities {a,b,c} and {x,y}
        return _FakeSeries([0, 0, 0, 1, 1]), 0.5

    fake_cugraph.Graph = _FakeGraph  # type: ignore[attr-defined]
    fake_cugraph.louvain = _fake_louvain  # type: ignore[attr-defined]

    edges = [
        CoPurchaseEdge("a", "b", 1.0),
        CoPurchaseEdge("b", "c", 1.0),
        CoPurchaseEdge("x", "y", 1.0),
    ]

    with (
        patch("lib.analytics.cugraph_optional._load_cugraph", return_value=fake_cugraph),
        patch("lib.analytics.cugraph_optional._CUDF_MODULE") as mock_cudf,
    ):
        mock_cudf.DataFrame = _FakeDataFrame
        communities = detect_louvain_communities_gpu(edges)

    assert communities is not None
    assert len(communities) == 2
    ids = {frozenset(community) for community in communities}
    assert frozenset({"a", "b", "c"}) in ids
    assert frozenset({"x", "y"}) in ids


def test_cugraph_available_false_on_import_error() -> None:
    import lib.analytics.cugraph_optional as module

    module._CUGRAPH_IMPORT_ATTEMPTED = False
    module._CUGRAPH_MODULE = None
    module._CUDF_MODULE = None

    with (
        patch("lib.analytics.cugraph_optional.is_cuda_available", return_value=True),
        patch(
            "lib.analytics.cugraph_optional.importlib.import_module",
            side_effect=ImportError("no cugraph"),
        ),
    ):
        assert cugraph_available() is False
