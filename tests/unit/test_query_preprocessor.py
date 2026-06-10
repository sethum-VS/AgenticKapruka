"""Unit tests for lib.chat.query_preprocessor (HybridRAG input pre-processing)."""

from __future__ import annotations

from lib.chat.query_preprocessor import (
    classify_query_mode,
    detect_code_switching,
    vernacular_score_hint,
)


def test_classify_utility_for_transactional_search() -> None:
    assert classify_query_mode("Show me birthday cakes under 5000 rupees") == "utility"


def test_classify_situational_for_breakup() -> None:
    message = "I broke up with my girlfriend and feel heartbroken"
    assert classify_query_mode(message) == "situational"


def test_detect_tanglish_code_switching() -> None:
    assert detect_code_switching("Mage ammata cake ekak ona") is True


def test_detect_plain_english_not_code_switched() -> None:
    assert detect_code_switching("Birthday cake for my mom in Colombo") is False


def test_vernacular_hint_higher_for_tanglish() -> None:
    tanglish = vernacular_score_hint("Machan, mage ammata cake ekak")
    formal = vernacular_score_hint("Please assist with a birthday cake order.")
    assert tanglish > formal
