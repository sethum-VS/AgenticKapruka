#!/usr/bin/env python3
"""Batch-embed Occasion, Category, and ProductType ontology nodes in Neo4j."""

from __future__ import annotations

import asyncio
import sys

from app.config import get_settings
from lib.neo4j.client import Neo4jClient
from lib.neo4j.embed_ontology import (
    count_nodes_with_embedding,
    embed_ontology_nodes,
    has_category_embeddings,
)
from lib.neo4j.ontology import LABEL_CATEGORY, LABEL_OCCASION, LABEL_PRODUCT_TYPE


async def _run() -> int:
    settings = get_settings()
    client = await Neo4jClient.connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    try:
        stats = await embed_ontology_nodes(client)
        if not await has_category_embeddings(client):
            print("ERROR: no Category nodes with embedding after embed run", file=sys.stderr)
            return 1

        print(
            f"Embedded {stats.nodes_embedded} ontology nodes in {stats.batches_written} batch(es)."
        )
        for label in (LABEL_CATEGORY, LABEL_OCCASION, LABEL_PRODUCT_TYPE):
            count = await count_nodes_with_embedding(client, label=label)
            print(f"  {label} with embedding: {count}")
        return 0
    finally:
        await client.close()


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
