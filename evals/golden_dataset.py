"""Pydantic schema and loader for the Ragas golden evaluation dataset."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

GoldenScenario = Literal["discovery", "checkout", "tracking"]

KNOWN_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "kapruka_search_products",
        "kapruka_get_product",
        "kapruka_list_categories",
        "kapruka_list_delivery_cities",
        "kapruka_check_delivery",
        "kapruka_create_order",
        "kapruka_track_order",
    }
)

DEFAULT_GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


class GoldenCase(BaseModel):
    """Single golden evaluation row for Ragas faithfulness and tool-routing checks."""

    id: str = Field(min_length=1, max_length=80)
    scenario: GoldenScenario
    user_query: str = Field(min_length=1, max_length=500)
    expected_tools: list[str]
    reference_answer: str = Field(min_length=1, max_length=2000)

    @field_validator("expected_tools")
    @classmethod
    def validate_expected_tools(cls, tools: list[str]) -> list[str]:
        unknown = [tool for tool in tools if tool not in KNOWN_MCP_TOOLS]
        if unknown:
            msg = f"Unknown MCP tool(s) in expected_tools: {', '.join(unknown)}"
            raise ValueError(msg)
        return tools


class GoldenDataset(BaseModel):
    """Top-level golden dataset envelope."""

    version: str = Field(min_length=1, max_length=20)
    cases: list[GoldenCase] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def validate_unique_ids(cls, cases: list[GoldenCase]) -> list[GoldenCase]:
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            msg = "Golden case ids must be unique"
            raise ValueError(msg)
        return cases


def load_golden_dataset(path: Path | None = None) -> GoldenDataset:
    """Load and validate golden_dataset.json from disk."""
    dataset_path = path or DEFAULT_GOLDEN_DATASET_PATH
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    return GoldenDataset.model_validate(raw)


@lru_cache(maxsize=1)
def get_golden_dataset() -> GoldenDataset:
    """Cached loader for tests and eval runners."""
    return load_golden_dataset()
