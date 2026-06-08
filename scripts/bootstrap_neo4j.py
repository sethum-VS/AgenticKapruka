#!/usr/bin/env python3
"""One-shot Neo4j bootstrap: schema → ingest → embed → vector index."""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.service import KaprukaService
from lib.neo4j.client import Neo4jClient
from lib.neo4j.embed_ontology import (
    count_nodes_with_embedding,
    embed_ontology_nodes,
    has_category_embeddings,
)
from lib.neo4j.ingest_categories import (
    INGEST_CATEGORY_DEPTH,
    count_ontology_nodes_by_label,
    ingest_category_tree,
)
from lib.neo4j.ontology import (
    LABEL_CATEGORY,
    apply_ontology_schema,
    list_ontology_constraints,
    verify_ontology_schema,
)
from lib.neo4j.vector_search import (
    create_category_vector_index,
    has_category_vector_index,
)
from lib.redis.client import RedisClient

_INGEST_CLIENT_IP = "127.0.0.1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-migrate",
        action="store_true",
        help="Skip ontology constraint migration",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip Kapruka category tree ingest",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip Vertex embedding of ontology nodes",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip Category vector index creation",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=INGEST_CATEGORY_DEPTH,
        choices=(1, 2),
        help="Kapruka category tree depth for ingest (default 2)",
    )
    return parser.parse_args()


async def _run_migrate(client: Neo4jClient) -> bool:
    await apply_ontology_schema(client)
    if not await verify_ontology_schema(client):
        print("ERROR: ontology constraints missing after migration", file=sys.stderr)
        return False
    rows = await list_ontology_constraints(client)
    print(f"Ontology schema OK ({len(rows)} constraints verified).")
    return True


async def _run_ingest(client: Neo4jClient, depth: int) -> bool:
    settings = get_settings()
    redis = await RedisClient.connect(settings.redis_url)
    mcp = await MCPHttpClient.connect(settings.kapruka_mcp_url)
    try:
        service = KaprukaService(redis, mcp)
        categories_output = await service.list_categories(_INGEST_CLIENT_IP, depth=depth)

        await apply_ontology_schema(client)
        if not await verify_ontology_schema(client):
            print("ERROR: ontology constraints missing before ingest", file=sys.stderr)
            return False

        stats = await ingest_category_tree(client, categories_output.categories)
        counts = await count_ontology_nodes_by_label(client)
        print(
            f"Ingested {stats.triplets_merged} triplets "
            f"and {stats.categories_merged} standalone categories "
            f"(depth={depth})."
        )
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
        return True
    finally:
        await mcp.close()
        await redis.close()


async def _run_embed(client: Neo4jClient) -> bool:
    stats = await embed_ontology_nodes(client)
    if not await has_category_embeddings(client):
        print("ERROR: no Category nodes with embedding after embed", file=sys.stderr)
        return False
    print(f"Embedded {stats.nodes_embedded} ontology nodes in {stats.batches_written} batch(es).")
    return True


async def _run_index(client: Neo4jClient) -> bool:
    await create_category_vector_index(client)
    if not await has_category_vector_index(client):
        print("ERROR: vector index missing after create", file=sys.stderr)
        return False
    print("Vector index ontology_category_embedding OK.")
    return True


async def _verify(client: Neo4jClient) -> bool:
    if not await has_category_embeddings(client):
        print("ERROR: verification failed — no category embeddings", file=sys.stderr)
        return False
    if not await has_category_vector_index(client):
        print("ERROR: verification failed — no vector index", file=sys.stderr)
        return False
    count = await count_nodes_with_embedding(client, label=LABEL_CATEGORY)
    print(f"Verification OK: {count} Category nodes with embeddings, vector index present.")
    return True


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    client = await Neo4jClient.connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    try:
        if not args.skip_migrate and not await _run_migrate(client):
            return 1
        if not args.skip_ingest and not await _run_ingest(client, args.depth):
            return 1
        if not args.skip_embed and not await _run_embed(client):
            return 1
        if not args.skip_index and not await _run_index(client):
            return 1
        if not await _verify(client):
            return 1
        print("Neo4j bootstrap complete.")
        return 0
    finally:
        await client.close()


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
