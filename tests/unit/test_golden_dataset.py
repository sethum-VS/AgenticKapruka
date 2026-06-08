"""Validate evals/golden_dataset.json schema and PRD coverage requirements."""

from __future__ import annotations

from pathlib import Path

import pytest
from evals.golden_dataset import (
    DEFAULT_GOLDEN_DATASET_PATH,
    GoldenDataset,
    load_golden_dataset,
)

pytestmark = pytest.mark.usefixtures("golden_dataset")


@pytest.fixture
def golden_dataset() -> GoldenDataset:
    """Load the committed golden dataset from evals/golden_dataset.json."""
    return load_golden_dataset()


@pytest.fixture
def golden_dataset_path() -> Path:
    """Path to the golden dataset JSON file."""
    return DEFAULT_GOLDEN_DATASET_PATH


def test_golden_dataset_file_exists(golden_dataset_path: Path) -> None:
    assert golden_dataset_path.is_file()


def test_golden_dataset_has_minimum_case_count(golden_dataset: GoldenDataset) -> None:
    assert len(golden_dataset.cases) >= 20


def test_golden_dataset_required_fields_on_every_case(golden_dataset: GoldenDataset) -> None:
    for case in golden_dataset.cases:
        assert case.user_query.strip()
        assert case.reference_answer.strip()
        assert isinstance(case.expected_tools, list)


def test_golden_dataset_covers_all_scenarios(golden_dataset: GoldenDataset) -> None:
    scenarios = {case.scenario for case in golden_dataset.cases}
    assert scenarios == {"discovery", "checkout", "tracking"}


def test_golden_dataset_birthday_cake_search_case(golden_dataset: GoldenDataset) -> None:
    case = next(c for c in golden_dataset.cases if c.id == "disc-001-birthday-cake-mom")
    assert case.scenario == "discovery"
    assert "birthday" in case.user_query.lower()
    assert "kapruka_search_products" in case.expected_tools


def test_golden_dataset_flower_delivery_check_case(golden_dataset: GoldenDataset) -> None:
    case = next(c for c in golden_dataset.cases if c.id == "checkout-001-flower-delivery-colombo")
    assert case.scenario == "checkout"
    assert "flower" in case.user_query.lower() or "colombo" in case.user_query.lower()
    assert "kapruka_check_delivery" in case.expected_tools


def test_golden_dataset_order_tracking_case(golden_dataset: GoldenDataset) -> None:
    case = next(c for c in golden_dataset.cases if c.id == "track-001-order-number")
    assert case.scenario == "tracking"
    assert "kapruka_track_order" in case.expected_tools
    assert "VIMP" in case.user_query


def test_golden_dataset_unique_case_ids(golden_dataset: GoldenDataset) -> None:
    ids = [case.id for case in golden_dataset.cases]
    assert len(ids) == len(set(ids))


def test_load_golden_dataset_rejects_unknown_tool(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        """{
  "version": "1",
  "cases": [{
    "id": "bad-001",
    "scenario": "discovery",
    "user_query": "test",
    "expected_tools": ["kapruka_unknown_tool"],
    "reference_answer": "answer"
  }]
}""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown MCP tool"):
        load_golden_dataset(bad_path)
