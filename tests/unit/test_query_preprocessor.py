"""Unit tests for lib.chat.query_preprocessor (HybridRAG input pre-processing)."""

from __future__ import annotations

from lib.chat.query_preprocessor import (
    QueryPreprocessor,
    classify_query_mode,
    detect_code_switching,
    detect_vernacular,
    extract_target_city,
    vernacular_score_hint,
)

_preprocessor = QueryPreprocessor()


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


def test_detect_vernacular_tanglish() -> None:
    assert detect_vernacular("Mage ammata cake ekak ona") == "tanglish"


def test_detect_vernacular_english() -> None:
    assert detect_vernacular("Birthday cake for my mom") == "en"


def test_extract_target_city_for_deliver_to_kandy() -> None:
    assert extract_target_city("Can you deliver to Kandy on Sunday?") == "Kandy"


def test_extract_target_city_colombo_zone_without_delivery_verb() -> None:
    assert extract_target_city("Birthday cake for my mom in Colombo 05") == "Colombo 05"
    assert extract_target_city("Birthday cake for my mom in Colombo") == "Colombo"


def test_query_preprocessor_extracts_colombo_zone_on_first_turn() -> None:
    metadata = _preprocessor.process("Birthday cake for my mom in Colombo 05")
    assert metadata["target_city"] == "Colombo 05"
    assert metadata["requires_delivery_validation"] is True


def test_query_preprocessor_utility_transactional() -> None:
    metadata = _preprocessor.process("Show me birthday cakes under 5000 rupees")
    assert metadata == {
        "is_situational": False,
        "detected_vernacular": "en",
        "requires_delivery_validation": False,
        "target_city": None,
        "budget_max": 5000.0,
    }


def test_query_preprocessor_tanglish_delivery_city() -> None:
    metadata = _preprocessor.process("Colombo delivery puluvan da?")
    assert metadata["is_situational"] is False
    assert metadata["detected_vernacular"] == "tanglish"
    assert metadata["requires_delivery_validation"] is True
    assert metadata["target_city"] == "Colombo"


def test_query_preprocessor_situational_breakup() -> None:
    metadata = _preprocessor.process("I broke up with my girlfriend and feel heartbroken")
    assert metadata["is_situational"] is True
    assert metadata["detected_vernacular"] == "en"
    assert metadata["requires_delivery_validation"] is False
    assert metadata["target_city"] is None


def test_query_preprocessor_perishable_gift_with_city_requires_delivery() -> None:
    metadata = _preprocessor.process("Birthday cake for my mom in Colombo")
    assert metadata["target_city"] == "Colombo"
    assert metadata["requires_delivery_validation"] is True


def test_classify_situational_for_valentine_nerves() -> None:
    message = "Valentine's surprise for my partner — I'm nervous"
    assert classify_query_mode(message) == "situational"
    assert _preprocessor.process(message)["is_situational"] is True
