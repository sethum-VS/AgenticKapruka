#!/usr/bin/env python3
"""Generate Vertex AI SFT .jsonl from Neo4j ontology + Sri Lankan vernacular templates."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from app.config import get_settings
from lib.neo4j.client import Neo4jClient
from lib.neo4j.ontology import LABEL_CATEGORY, LABEL_OCCASION

Role = Literal["user", "model"]

# Colloquial Sri Lankan terms → Kapruka category slug (ingest_categories.slugify_name).
VERNAULAR_TEMPLATES: Final[dict[str, str]] = {
    "poya day": "cakes",
    "kiri bath": "groceries",
    "baila party": "gift-items",
    "sinhala new year": "cakes",
    "avurudu": "cakes",
    "aurudu": "cakes",
    "vesak": "cakes",
    "poson": "cakes",
    "tamil new year": "cakes",
    "festive season": "gift-hampers",
    "wedding": "flowers",
    "api adare": "chocolates",
}

USER_UTTERANCE_PATTERNS: Final[tuple[str, ...]] = (
    "Need something for {vernacular} delivery to Colombo",
    "{vernacular} gifts for family please",
    "mama {vernacular} cake ona eka send karanna",
    "looking for {vernacular} {category_hint} ideas",
    "can you help with {vernacular} {occasion_hint} shopping",
)

_FETCH_ONTOLOGY_CYPHER = f"""
MATCH (n)
WHERE n:{LABEL_CATEGORY} OR n:{LABEL_OCCASION}
RETURN
    labels(n)[0] AS label,
    n.slug AS slug,
    n.display_name AS display_name,
    n.description AS description
ORDER BY label, slug
""".strip()

_FETCH_OCCASIONS_FOR_CATEGORY_CYPHER = f"""
MATCH (o:{LABEL_OCCASION})-[:OCCASION_TO_CATEGORY]->(c:{LABEL_CATEGORY})
WHERE c.slug = $category_slug
RETURN
    o.slug AS slug,
    o.display_name AS display_name,
    o.description AS description
ORDER BY o.display_name
LIMIT 3
""".strip()

DEFAULT_OUTPUT_PATH = Path("data/lora_sft_dataset.jsonl")


@dataclass(frozen=True)
class OntologyNode:
    """Lightweight Occasion or Category row from Neo4j."""

    label: str
    slug: str
    display_name: str
    description: str | None = None


@dataclass(frozen=True)
class SftMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class SftRecord:
    messages: tuple[SftMessage, SftMessage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": self.messages[0].role, "content": self.messages[0].content},
                {"role": self.messages[1].role, "content": self.messages[1].content},
            ]
        }


def _slug_to_display_name(slug: str) -> str:
    return slug.replace("-", " ").title()


def _resolve_category(nodes: dict[str, OntologyNode], category_slug: str) -> OntologyNode:
    found = nodes.get(category_slug)
    if found is not None:
        return found
    return OntologyNode(
        label=LABEL_CATEGORY,
        slug=category_slug,
        display_name=_slug_to_display_name(category_slug),
    )


async def fetch_ontology_nodes(client: Neo4jClient) -> dict[str, OntologyNode]:
    """Load Category and Occasion nodes keyed by slug."""
    rows = await client.execute(_FETCH_ONTOLOGY_CYPHER)
    nodes: dict[str, OntologyNode] = {}
    for row in rows:
        slug = row.get("slug")
        display_name = row.get("display_name")
        if not slug or not display_name:
            continue
        nodes[str(slug)] = OntologyNode(
            label=str(row.get("label") or ""),
            slug=str(slug),
            display_name=str(display_name),
            description=row.get("description"),
        )
    return nodes


async def fetch_occasions_for_category(
    client: Neo4jClient,
    category_slug: str,
) -> list[OntologyNode]:
    """Return Occasions linked to a Category via OCCASION_TO_CATEGORY."""
    rows = await client.execute(
        _FETCH_OCCASIONS_FOR_CATEGORY_CYPHER,
        {"category_slug": category_slug},
    )
    occasions: list[OntologyNode] = []
    for row in rows:
        slug = row.get("slug")
        display_name = row.get("display_name")
        if not slug or not display_name:
            continue
        occasions.append(
            OntologyNode(
                label=LABEL_OCCASION,
                slug=str(slug),
                display_name=str(display_name),
                description=row.get("description"),
            )
        )
    return occasions


def _build_model_content(
    *,
    vernacular: str,
    category_display: str,
    occasion_display: str | None,
) -> str:
    """Structured discovery-args JSON for the model turn (q + category)."""
    if occasion_display:
        q = f"{vernacular} {occasion_display} {category_display}".strip()
    else:
        q = f"{vernacular} {category_display}".strip()
    payload = {"q": " ".join(q.split()), "category": category_display}
    return json.dumps(payload, ensure_ascii=False)


def build_training_pairs(
    *,
    ontology_by_slug: dict[str, OntologyNode],
    occasions_by_category: dict[str, list[OntologyNode]],
) -> list[SftRecord]:
    """Combine vernacular templates with ontology enrichment into SFT pairs."""
    records: list[SftRecord] = []

    for vernacular, category_slug in VERNAULAR_TEMPLATES.items():
        category = _resolve_category(ontology_by_slug, category_slug)
        linked_occasions = occasions_by_category.get(category_slug, [])
        occasion = linked_occasions[0] if linked_occasions else None

        category_hint = category.display_name.lower()
        occasion_hint = occasion.display_name.lower() if occasion else category_hint

        for pattern in USER_UTTERANCE_PATTERNS:
            user_content = pattern.format(
                vernacular=vernacular,
                category_hint=category_hint,
                occasion_hint=occasion_hint,
            )
            model_content = _build_model_content(
                vernacular=vernacular,
                category_display=category.display_name,
                occasion_display=occasion.display_name if occasion else None,
            )
            records.append(
                SftRecord(
                    messages=(
                        SftMessage(role="user", content=user_content),
                        SftMessage(role="model", content=model_content),
                    )
                )
            )

    return records


def validate_sft_record(record: dict[str, Any]) -> None:
    """Validate a single Vertex AI conversational SFT JSONL record."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        msg = "record must have exactly two messages"
        raise ValueError(msg)

    user_msg, model_msg = messages
    if user_msg.get("role") != "user":
        msg = "first message must be role=user"
        raise ValueError(msg)
    if model_msg.get("role") != "model":
        msg = "second message must be role=model"
        raise ValueError(msg)

    user_content = user_msg.get("content")
    model_content = model_msg.get("content")
    if not isinstance(user_content, str) or not user_content.strip():
        msg = "user content must be a non-empty string"
        raise ValueError(msg)
    if not isinstance(model_content, str) or not model_content.strip():
        msg = "model content must be a non-empty string"
        raise ValueError(msg)

    try:
        payload = json.loads(model_content)
    except json.JSONDecodeError as exc:
        msg = f"model content is not valid JSON: {model_content!r}"
        raise ValueError(msg) from exc

    if not isinstance(payload, dict):
        msg = "model JSON must be an object"
        raise ValueError(msg)

    q = payload.get("q")
    category = payload.get("category")
    if not isinstance(q, str) or not q.strip():
        msg = "model JSON must include non-empty string q"
        raise ValueError(msg)
    if not isinstance(category, str) or not category.strip():
        msg = "model JSON must include non-empty string category"
        raise ValueError(msg)


def validate_dataset(records: list[dict[str, Any]]) -> None:
    """Validate all records and require at least one Avurudu/Festive example."""
    if not records:
        msg = "dataset must contain at least one training pair"
        raise ValueError(msg)

    for record in records:
        validate_sft_record(record)

    serialized = json.dumps(records, ensure_ascii=False).lower()
    if "avurudu" not in serialized and "aurudu" not in serialized and "festive" not in serialized:
        msg = "dataset must include at least one Avurudu or Festive example pair"
        raise ValueError(msg)


def write_jsonl(path: Path, records: list[SftRecord]) -> int:
    """Write SFT records as JSONL; returns line count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dict_records = [record.to_dict() for record in records]
    validate_dataset(dict_records)
    with path.open("w", encoding="utf-8") as handle:
        for record in dict_records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return len(dict_records)


async def generate_dataset(client: Neo4jClient) -> list[SftRecord]:
    """Query Neo4j and build the full training set."""
    ontology_by_slug = await fetch_ontology_nodes(client)
    occasions_by_category: dict[str, list[OntologyNode]] = {}
    for category_slug in set(VERNAULAR_TEMPLATES.values()):
        occasions_by_category[category_slug] = await fetch_occasions_for_category(
            client,
            category_slug,
        )
    return build_training_pairs(
        ontology_by_slug=ontology_by_slug,
        occasions_by_category=occasions_by_category,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output .jsonl path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Build with mocked/in-memory ontology and validate schema only",
    )
    return parser.parse_args()


def _self_check_records() -> list[SftRecord]:
    """Build a minimal dataset without Neo4j for schema validation."""
    ontology = {
        "cakes": OntologyNode(
            label=LABEL_CATEGORY,
            slug="cakes",
            display_name="Cakes",
            description="Kapruka cakes and sweets",
        ),
        "gift-hampers": OntologyNode(
            label=LABEL_CATEGORY,
            slug="gift-hampers",
            display_name="Gift Hampers",
            description="Festive gift hampers",
        ),
    }
    occasions = {
        "cakes": [
            OntologyNode(
                label=LABEL_OCCASION,
                slug="birthday",
                display_name="Birthday",
            )
        ],
        "gift-hampers": [
            OntologyNode(
                label=LABEL_OCCASION,
                slug="festive",
                display_name="Festive",
            )
        ],
    }
    return build_training_pairs(ontology_by_slug=ontology, occasions_by_category=occasions)


async def _run(args: argparse.Namespace) -> int:
    if args.self_check:
        records = _self_check_records()
        count = write_jsonl(args.output, records)
        print(f"Self-check OK: wrote {count} validated record(s) to {args.output}")
        return 0

    settings = get_settings()
    client = await Neo4jClient.connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    try:
        records = await generate_dataset(client)
        count = write_jsonl(args.output, records)
    finally:
        await client.close()

    print(f"Wrote {count} LoRA SFT record(s) to {args.output}")
    return 0


def main() -> None:
    args = _parse_args()
    try:
        code = asyncio.run(_run(args))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
