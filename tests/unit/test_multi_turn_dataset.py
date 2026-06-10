"""Validate evals/multi_turn_dataset.json for shadow-test coverage."""

from __future__ import annotations

from evals.multi_turn_dataset import get_multi_turn_dataset, load_multi_turn_dataset


def test_multi_turn_dataset_loads() -> None:
    dataset = load_multi_turn_dataset()
    assert dataset.version == "1"


def test_multi_turn_dataset_minimum_cases() -> None:
    """Target is 50 personas; MVP gate at 10 until dataset is expanded."""
    dataset = get_multi_turn_dataset()
    assert len(dataset.cases) >= 10


def test_multi_turn_cases_have_unique_ids() -> None:
    dataset = get_multi_turn_dataset()
    ids = [case.id for case in dataset.cases]
    assert len(ids) == len(set(ids))


def test_multi_turn_includes_situational_persona() -> None:
    modes = {case.query_mode for case in get_multi_turn_dataset().cases}
    assert "situational" in modes


def test_multi_turn_includes_vernacular_persona() -> None:
    personas = {case.persona for case in get_multi_turn_dataset().cases}
    assert "vernacular_user" in personas
