#!/usr/bin/env python3
"""Bootstrap Neo4j GraphRAG ontology: schema, ingest, embed, and vector indexes."""

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
    LABEL_OCCASION,
    LABEL_PRODUCT_TYPE,
    apply_ontology_schema,
    verify_ontology_schema,
)
from lib.neo4j.vector_search import (
    create_ontology_vector_indexes,
    has_category_vector_index,
    has_occasion_vector_index,
)
from lib.redis.client import RedisClient

_BOOTSTRAP_CLIENT_IP = "127.0.0.1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth",
        type=int,
        default=INGEST_CATEGORY_DEPTH,
        choices=(1, 2),
        help="Kapruka category tree depth (MCP supports 1–2; default 2 for triplets)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip Kapruka category ingest (schema, embed, and indexes only)",
    )
    return parser.parse_args()


async def _run_schema(client: Neo4jClient) -> bool:
    await apply_ontology_schema(client)
    if not await verify_ontology_schema(client):
        print("ERROR: ontology constraints missing after schema migration", file=sys.stderr)
        return False
    print("Ontology schema applied.")
    return True


async def _run_ingest(
    client: Neo4jClient,
    service: KaprukaService,
    *,
    depth: int,
) -> bool:
    categories_output = await service.list_categories(_BOOTSTRAP_CLIENT_IP, depth=depth)
    stats = await ingest_category_tree(client, categories_output.categories)
    counts = await count_ontology_nodes_by_label(client)
    print(
        f"Ingested {stats.triplets_merged} triplets "
        f"and {stats.categories_merged} standalone categories (depth={depth})."
    )
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    return True


async def _run_embed(client: Neo4jClient) -> bool:
    stats = await embed_ontology_nodes(client)
    if not await has_category_embeddings(client):
        print("ERROR: no Category nodes with embedding after embed run", file=sys.stderr)
        return False
    print(f"Embedded {stats.nodes_embedded} ontology nodes in {stats.batches_written} batch(es).")
    for label in (LABEL_CATEGORY, LABEL_OCCASION, LABEL_PRODUCT_TYPE):
        count = await count_nodes_with_embedding(client, label=label)
        print(f"  {label} with embedding: {count}")
    return True


async def _run_index(client: Neo4jClient) -> bool:
    await create_ontology_vector_indexes(client)
    if not await has_category_vector_index(client):
        print("ERROR: Category vector index missing after bootstrap", file=sys.stderr)
        return False
    if not await has_occasion_vector_index(client):
        print("ERROR: Occasion vector index missing after bootstrap", file=sys.stderr)
        return False
    print("Ontology vector indexes created (Category, Occasion).")
    return True


async def _run(*, depth: int, skip_ingest: bool) -> int:
    settings = get_settings()
    redis = await RedisClient.connect(settings.redis_url)
    mcp = await MCPHttpClient.connect(settings.kapruka_mcp_url)
    client = await Neo4jClient.connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    try:
        service = KaprukaService(redis, mcp)
        if not await _run_schema(client):
            return 1
        if not skip_ingest and not await _run_ingest(client, service, depth=depth):
            return 1
        if not await _run_embed(client):
            return 1
        if not await _run_index(client):
            return 1
        print("Neo4j bootstrap complete.")
        return 0
    finally:
        await client.close()
        await mcp.close()
        await redis.close()


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(depth=args.depth, skip_ingest=args.skip_ingest)))


if __name__ == "__main__":
    main()
