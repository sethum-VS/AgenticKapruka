#!/usr/bin/env python3
"""Apply Neo4j GraphRAG ontology constraints (Occasion, Category, ProductType)."""

from __future__ import annotations

import asyncio
import sys

from app.config import get_settings
from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import (
    apply_ontology_schema,
    list_ontology_constraints,
    verify_ontology_schema,
)


async def _run() -> int:
    settings = get_settings()
    client = await Neo4jClient.connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    try:
        await apply_ontology_schema(client)
        if not await verify_ontology_schema(client):
            print("ERROR: ontology constraints missing after migration", file=sys.stderr)
            return 1
        rows = await list_ontology_constraints(client)
        for row in rows:
            print(f"  {row['name']}: {row['type']} on {row['labelsOrTypes']} ({row['properties']})")
        print(f"Ontology schema OK ({len(rows)} constraints verified).")
        return 0
    finally:
        await client.close()


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
