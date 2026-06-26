"""Unit tests for incremental agent-loop status copy."""

from __future__ import annotations

from lib.chat.status_copy import (
    CHECKING_DELIVERY,
    CURATING_FOR_BUDGET,
    PUTTING_TOGETHER_RECOMMENDATIONS,
    SEARCHING_CATALOG,
    SEARCHING_KAPRUKA,
    long_search_status_message,
)


def test_long_search_status_message_rotates_by_iteration() -> None:
    assert long_search_status_message(iteration=0) == SEARCHING_CATALOG
    assert long_search_status_message(iteration=1) == SEARCHING_KAPRUKA
    assert long_search_status_message(iteration=2) == CHECKING_DELIVERY


def test_long_search_status_message_budget_override_on_later_iterations() -> None:
    assert long_search_status_message(iteration=2, has_budget=True) == CURATING_FOR_BUDGET
    assert long_search_status_message(iteration=4, has_budget=False) == (
        PUTTING_TOGETHER_RECOMMENDATIONS
    )
