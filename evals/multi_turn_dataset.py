"""Schema and loader for multi-turn shadow-test transcripts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from evals.golden_dataset import KNOWN_MCP_TOOLS

DEFAULT_MULTI_TURN_PATH = Path(__file__).resolve().parent / "multi_turn_dataset.json"

PersonaTag = Literal[
    "busy_parent",
    "expat_sender",
    "emotional_gifter",
    "corporate_buyer",
    "vernacular_user",
    "budget_shopper",
]


class MultiTurnStep(BaseModel):
    """One user utterance in a shadow transcript."""

    role: Literal["user"] = "user"
    content: str = Field(min_length=1, max_length=500)


class PreservationConstraints(BaseModel):
    """Terms and tools that must survive to the final turn."""

    preserved_terms: list[str] = Field(default_factory=list)
    must_not_contain_in_final: list[str] = Field(default_factory=list)
    expected_tools_any_turn: list[str] = Field(default_factory=list)

    @field_validator("expected_tools_any_turn")
    @classmethod
    def validate_tools(cls, tools: list[str]) -> list[str]:
        unknown = [tool for tool in tools if tool not in KNOWN_MCP_TOOLS]
        if unknown:
            msg = f"Unknown MCP tool(s): {', '.join(unknown)}"
            raise ValueError(msg)
        return tools


class MultiTurnCase(BaseModel):
    """Parametrized real-world shopping persona with intent preservation checks."""

    id: str = Field(min_length=1, max_length=80)
    persona: PersonaTag
    query_mode: Literal["utility", "situational"] = "utility"
    turns: list[MultiTurnStep] = Field(min_length=2, max_length=8)
    constraints: PreservationConstraints = Field(default_factory=PreservationConstraints)
    final_expect_product_ui: bool = False
    final_expect_checkout_ui: bool = False

    @field_validator("turns")
    @classmethod
    def require_user_turns(cls, turns: list[MultiTurnStep]) -> list[MultiTurnStep]:
        if not turns:
            msg = "Multi-turn case requires at least one user turn"
            raise ValueError(msg)
        return turns


class MultiTurnDataset(BaseModel):
    version: str = Field(min_length=1, max_length=20)
    cases: list[MultiTurnCase] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def unique_ids(cls, cases: list[MultiTurnCase]) -> list[MultiTurnCase]:
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            msg = "Multi-turn case ids must be unique"
            raise ValueError(msg)
        return cases


def load_multi_turn_dataset(path: Path | None = None) -> MultiTurnDataset:
    dataset_path = path or DEFAULT_MULTI_TURN_PATH
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    return MultiTurnDataset.model_validate(raw)


@lru_cache(maxsize=1)
def get_multi_turn_dataset() -> MultiTurnDataset:
    return load_multi_turn_dataset()
