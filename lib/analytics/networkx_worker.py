"""NetworkX Louvain community detection background worker for co-purchase recommendations."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Final, Literal

import networkx as nx
from networkx.algorithms.community import louvain_communities

from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import (
    LABEL_CATEGORY,
    LABEL_PRODUCT_TYPE,
    REL_CATEGORY_TO_PRODUCT_TYPE,
)

logger = logging.getLogger(__name__)

REL_CO_PURCHASED_WITH: Final = "CO_PURCHASED_WITH"
REL_RECOMMENDS: Final = "RECOMMENDS"
LABEL_PRODUCT: Final = "Product"

DEFAULT_INTERVAL_SECONDS: Final = 3600
_LOUVAIN_SEED: Final = 42
_RECOMMENDS_BATCH_SIZE: Final = 100
_DEFAULT_COMMUNITY_SCORE: Final = 0.5

EdgeSource = Literal["co_purchase", "category_proximity"]

_FETCH_CO_PURCHASE_CYPHER = f"""
MATCH (a:{LABEL_PRODUCT})-[r:{REL_CO_PURCHASED_WITH}]->(b:{LABEL_PRODUCT})
RETURN a.id AS source_id, b.id AS target_id, coalesce(r.weight, 1.0) AS weight
""".strip()

_SYNTHESIZE_CATEGORY_PROXIMITY_CYPHER = f"""
MATCH (c:{LABEL_CATEGORY})-[r1:{REL_CATEGORY_TO_PRODUCT_TYPE}]->(a:{LABEL_PRODUCT_TYPE}),
      (c)-[r2:{REL_CATEGORY_TO_PRODUCT_TYPE}]->(b:{LABEL_PRODUCT_TYPE})
WHERE a.id < b.id
RETURN a.id AS source_id,
       b.id AS target_id,
       coalesce(r1.weight, 1.0) * coalesce(r2.weight, 1.0) AS weight
""".strip()

_MERGE_RECOMMENDS_BATCH_CYPHER = f"""
UNWIND $batch AS row
MATCH (a {{id: row.source_id}})
MATCH (b {{id: row.target_id}})
WHERE a.id <> b.id
MERGE (a)-[r:{REL_RECOMMENDS}]->(b)
SET r.score = row.score,
    r.community_id = row.community_id,
    r.source = row.source
""".strip()


@dataclass(frozen=True, slots=True)
class CoPurchaseEdge:
    """Weighted undirected co-purchase or proximity edge between two node ids."""

    source_id: str
    target_id: str
    weight: float


@dataclass(frozen=True, slots=True)
class CommunityDetectionResult:
    """Summary of a single community-detection run."""

    communities_found: int
    recommends_written: int
    nodes_in_graph: int
    edge_count: int
    edge_source: EdgeSource


async def fetch_co_purchase_edges(client: Neo4jClient) -> list[CoPurchaseEdge]:
    """Load observed Product co-purchase edges from Neo4j."""
    rows = await client.execute(_FETCH_CO_PURCHASE_CYPHER)
    return [_edge_from_row(row) for row in rows]


async def synthesize_category_proximity_edges(client: Neo4jClient) -> list[CoPurchaseEdge]:
    """Synthesize edges between ProductTypes that share a parent Category."""
    rows = await client.execute(_SYNTHESIZE_CATEGORY_PROXIMITY_CYPHER)
    return [_edge_from_row(row) for row in rows]


def _edge_from_row(row: dict[str, Any]) -> CoPurchaseEdge:
    return CoPurchaseEdge(
        source_id=str(row["source_id"]),
        target_id=str(row["target_id"]),
        weight=float(row.get("weight", 1.0)),
    )


def build_networkx_graph(edges: list[CoPurchaseEdge]) -> nx.Graph:
    """Build an undirected weighted NetworkX graph from co-purchase edges."""
    graph: nx.Graph = nx.Graph()
    for edge in edges:
        graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
    return graph


def detect_louvain_communities(graph: nx.Graph) -> list[frozenset[str]]:
    """Run Louvain community detection (CPU-bound; call via thread executor)."""
    if graph.number_of_nodes() == 0:
        return []
    partition = louvain_communities(graph, weight="weight", seed=_LOUVAIN_SEED)
    return [frozenset(community) for community in partition]


def build_recommendation_rows(
    communities: list[frozenset[str]],
    edges: list[CoPurchaseEdge],
    *,
    edge_source: EdgeSource,
) -> list[dict[str, Any]]:
    """Build RECOMMENDS rows for same-community pairs with edge or default score."""
    direct_weights: dict[frozenset[str], float] = {}
    for edge in edges:
        key = frozenset((edge.source_id, edge.target_id))
        direct_weights[key] = max(direct_weights.get(key, 0.0), edge.weight)

    rows: list[dict[str, Any]] = []
    for community_index, community in enumerate(communities):
        if len(community) < 2:
            continue
        community_id = str(community_index)
        members = sorted(community)
        for index, source_id in enumerate(members):
            for target_id in members[index + 1 :]:
                pair_key = frozenset((source_id, target_id))
                score = direct_weights.get(pair_key, _DEFAULT_COMMUNITY_SCORE)
                for from_id, to_id in ((source_id, target_id), (target_id, source_id)):
                    rows.append(
                        {
                            "source_id": from_id,
                            "target_id": to_id,
                            "score": score,
                            "community_id": community_id,
                            "source": edge_source,
                        }
                    )
    return rows


async def persist_recommends(
    client: Neo4jClient,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = _RECOMMENDS_BATCH_SIZE,
) -> int:
    """MERGE RECOMMENDS relationships in batches; return rows written."""
    written = 0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        if not batch:
            continue
        await client.execute(_MERGE_RECOMMENDS_BATCH_CYPHER, {"batch": batch})
        written += len(batch)
    return written


async def run_community_detection(client: Neo4jClient) -> CommunityDetectionResult:
    """Export edges, detect communities, and persist RECOMMENDS to Neo4j."""
    edges = await fetch_co_purchase_edges(client)
    edge_source: EdgeSource = "co_purchase"
    if not edges:
        edges = await synthesize_category_proximity_edges(client)
        edge_source = "category_proximity"

    graph = build_networkx_graph(edges)
    communities = await asyncio.to_thread(detect_louvain_communities, graph)
    recommendation_rows = build_recommendation_rows(
        communities,
        edges,
        edge_source=edge_source,
    )
    recommends_written = await persist_recommends(client, recommendation_rows)

    result = CommunityDetectionResult(
        communities_found=len(communities),
        recommends_written=recommends_written,
        nodes_in_graph=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        edge_source=edge_source,
    )
    logger.info(
        "Community detection complete: %s communities, %s RECOMMENDS, source=%s",
        result.communities_found,
        result.recommends_written,
        result.edge_source,
    )
    return result


class NetworkXCommunityWorker:
    """Periodic asyncio background worker for Louvain community detection."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        *,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self._neo4j = neo4j_client
        self._interval_seconds = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def run_once(self) -> CommunityDetectionResult:
        """Run a single community-detection cycle."""
        return await run_community_detection(self._neo4j)

    async def start(self) -> None:
        """Start the background loop; no-op if already running."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(), name="networkx-community-worker")
        logger.info(
            "NetworkX community worker started (interval=%ss)",
            self._interval_seconds,
        )

    async def stop(self) -> None:
        """Cancel the background loop and wait for shutdown."""
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info("NetworkX community worker stopped")

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("Community detection cycle failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval_seconds,
                )
                break
            except TimeoutError:
                continue
