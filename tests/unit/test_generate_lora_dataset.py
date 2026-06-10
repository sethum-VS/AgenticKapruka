"""Tests for scripts/generate_lora_dataset.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from scripts import generate_lora_dataset as lora_script

from lib.neo4j.ontology import LABEL_CATEGORY, LABEL_OCCASION


def _sample_ontology() -> dict[str, lora_script.OntologyNode]:
    return {
        "cakes": lora_script.OntologyNode(
            label=LABEL_CATEGORY,
            slug="cakes",
            display_name="Cakes",
            description="Kapruka cakes",
        ),
        "flowers": lora_script.OntologyNode(
            label=LABEL_CATEGORY,
            slug="flowers",
            display_name="Flowers",
        ),
        "gift-hampers": lora_script.OntologyNode(
            label=LABEL_CATEGORY,
            slug="gift-hampers",
            display_name="Gift Hampers",
        ),
    }


def test_vernacular_templates_map_to_category_slugs() -> None:
    assert lora_script.VERNAULAR_TEMPLATES["avurudu"] == "cakes"
    assert lora_script.VERNAULAR_TEMPLATES["poya day"] == "cakes"
    assert lora_script.VERNAULAR_TEMPLATES["festive season"] == "gift-hampers"
    assert len(lora_script.VERNAULAR_TEMPLATES) >= 8


def test_build_training_pairs_emits_vertex_sft_schema() -> None:
    records = lora_script.build_training_pairs(
        ontology_by_slug=_sample_ontology(),
        occasions_by_category={
            "cakes": [
                lora_script.OntologyNode(
                    label=LABEL_OCCASION,
                    slug="birthday",
                    display_name="Birthday",
                )
            ],
            "gift-hampers": [
                lora_script.OntologyNode(
                    label=LABEL_OCCASION,
                    slug="festive",
                    display_name="Festive",
                )
            ],
        },
    )
    assert records
    for record in records:
        lora_script.validate_sft_record(record.to_dict())


def test_dataset_includes_avurudu_or_festive_example() -> None:
    records = lora_script.build_training_pairs(
        ontology_by_slug=_sample_ontology(),
        occasions_by_category={
            "cakes": [],
            "gift-hampers": [
                lora_script.OntologyNode(
                    label=LABEL_OCCASION,
                    slug="festive",
                    display_name="Festive",
                )
            ],
        },
    )
    payload = json.dumps([r.to_dict() for r in records], ensure_ascii=False).lower()
    assert "avurudu" in payload or "aurudu" in payload or "festive" in payload
    lora_script.validate_dataset([r.to_dict() for r in records])


def test_model_content_is_structured_discovery_json() -> None:
    records = lora_script.build_training_pairs(
        ontology_by_slug=_sample_ontology(),
        occasions_by_category={"cakes": []},
    )
    avurudu_record = next(r for r in records if "avurudu" in r.messages[0].content.lower())
    model_payload = json.loads(avurudu_record.messages[1].content)
    assert model_payload["category"] == "Cakes"
    assert "avurudu" in model_payload["q"].lower()


def test_write_jsonl_validates_and_writes(tmp_path: Path) -> None:
    records = lora_script._self_check_records()
    out = tmp_path / "lora.jsonl"
    count = lora_script.write_jsonl(out, records)
    assert count == len(records)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == count
    for line in lines:
        lora_script.validate_sft_record(json.loads(line))


def test_validate_sft_record_rejects_bad_model_json() -> None:
    bad = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "model", "content": "not-json"},
        ]
    }
    with pytest.raises(ValueError, match="not valid JSON"):
        lora_script.validate_sft_record(bad)


@pytest.mark.asyncio
async def test_generate_dataset_queries_neo4j() -> None:
    client = MagicMock()

    async def _execute(
        cypher: str,
        params: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        if "labels(n)[0]" in cypher:
            return [
                {
                    "label": LABEL_CATEGORY,
                    "slug": "cakes",
                    "display_name": "Cakes",
                    "description": "cakes",
                }
            ]
        if "category_slug" in (params or {}):
            if params["category_slug"] == "gift-hampers":
                return [{"slug": "festive", "display_name": "Festive", "description": None}]
            return [{"slug": "birthday", "display_name": "Birthday", "description": None}]
        return []

    client.execute = AsyncMock(side_effect=_execute)

    records = await lora_script.generate_dataset(client)
    assert records
    lora_script.validate_dataset([r.to_dict() for r in records])
    assert client.execute.await_count >= 1


@pytest.mark.asyncio
async def test_run_self_check_writes_output(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    args = argparse.Namespace(output=out, self_check=True)
    with patch.object(lora_script, "write_jsonl", wraps=lora_script.write_jsonl) as mock_write:
        code = await lora_script._run(args)
    assert code == 0
    mock_write.assert_called_once()
    assert out.is_file()
