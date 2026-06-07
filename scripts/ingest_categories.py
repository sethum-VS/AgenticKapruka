#!/usr/bin/env python3
"""Ingest Kapruka category taxonomy into Neo4j Occasion-Category-ProductType ontology."""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.service import KaprukaService
from lib.neo4j.client import Neo4jClient
from lib.neo4j.ingest_categories import (
    INGEST_CATEGORY_DEPTH,
    count_ontology_nodes_by_label,
    ingest_category_tree,
)
from lib.neo4j.ontology import apply_ontology_schema, verify_ontology_schema
from lib.redis.client import RedisClient

_INGEST_CLIENT_IP = "127.0.0.1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth",
        type=int,
        default=INGEST_CATEGORY_DEPTH,
        choices=(1, 2),
        help="Kapruka category tree depth (MCP supports 1–2; default 2 for triplets)",
    )
    return parser.parse_args()


async def _run(depth: int) -> int:
    settings = get_settings()
    redis = await RedisClient.connect(settings.redis_url)
    mcp = await MCPHttpClient.connect(settings.kapruka_mcp_url)
    try:
        service = KaprukaService(redis, mcp)
        categories_output = await service.list_categories(_INGEST_CLIENT_IP, depth=depth)

        neo4j = await Neo4jClient.connect(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )
        try:
            await apply_ontology_schema(neo4j)
            if not await verify_ontology_schema(neo4j):
                print("ERROR: ontology constraints missing before ingest", file=sys.stderr)
                return 1

            stats = await ingest_category_tree(neo4j, categories_output.categories)
            counts = await count_ontology_nodes_by_label(neo4j)

            print(
                f"Ingested {stats.triplets_merged} triplets "
                f"and {stats.categories_merged} standalone categories "
                f"(depth={depth})."
            )
            for label, count in sorted(counts.items()):
                print(f"  {label}: {count}")
            return 0
        finally:
            await neo4j.close()
    finally:
        await mcp.close()
        await redis.close()


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(args.depth)))


if __name__ == "__main__":
    main()
