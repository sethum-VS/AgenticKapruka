"""Unit tests for scripts/verify_chat_loop.py evaluation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_chat_loop.py"
_spec = importlib.util.spec_from_file_location("verify_chat_loop", _SCRIPT)
assert _spec and _spec.loader
_vcl = importlib.util.module_from_spec(_spec)
sys.modules["verify_chat_loop"] = _vcl
_spec.loader.exec_module(_vcl)

TurnScenario = _vcl.TurnScenario
_evaluate_turn = _vcl._evaluate_turn
_extract_top_carousel_card_texts = _vcl._extract_top_carousel_card_texts


def _carousel_html(*names: str, first_price: str = "4,500") -> str:
    cards = []
    for name in names:
        cards.append(
            f'<article data-testid="product-card"><h3>{name}</h3>'
            f'<p data-testid="product-price">Rs. {first_price}</p></article>'
        )
    return (
        '<div data-testid="product-carousel">'
        '<div data-testid="product-carousel-track">' + "".join(cards) + "</div></div>"
    )


def test_cake_mom_colombo_passes_carousel_without_api_errors() -> None:
    scenario = TurnScenario(
        name="cake_mom_colombo",
        message="Birthday cake for mom in Colombo",
        expect_carousel=False,
        expect_any_of=("carousel", "clarifying", "delivery"),
        forbidden_substrings=_vcl._API_ERROR_FORBIDDEN,
    )
    html = _carousel_html("Chocolate Birthday Cake")
    assert _evaluate_turn(scenario, html) == []


def test_cake_mom_colombo_fails_field_required() -> None:
    scenario = TurnScenario(
        name="cake_mom_colombo",
        message="Birthday cake for mom in Colombo",
        expect_carousel=False,
        expect_any_of=("carousel", "clarifying", "delivery"),
        forbidden_substrings=_vcl._API_ERROR_FORBIDDEN,
    )
    html = "<p>Field required for city</p>"
    failures = _evaluate_turn(scenario, html)
    assert any("Field required" in item for item in failures)


def test_gift_ideas_5000_requires_carousel_gift_and_budget() -> None:
    scenario = TurnScenario(
        name="gift_ideas_5000",
        message="Gift ideas under Rs. 5,000",
        expect_carousel=True,
        max_first_carousel_price=5000.0,
        expect_carousel_keywords=("gift", "voucher"),
        forbidden_substrings=_vcl._API_ERROR_FORBIDDEN,
    )
    ok_html = _carousel_html("Kapruka Gift Voucher", first_price="4,500")
    assert _evaluate_turn(scenario, ok_html) == []

    over_budget = _carousel_html("Gift Box", first_price="6,500")
    failures = _evaluate_turn(scenario, over_budget)
    assert any("exceeds budget" in item for item in failures)


def test_flowers_fruit_kandy_forbids_puja_in_top_slots() -> None:
    scenario = TurnScenario(
        name="flowers_fruit_kandy",
        message="flowers and fruit basket for Kandy on June 19, budget 5000 LKR",
        expect_carousel=True,
        max_first_carousel_price=5000.0,
        forbidden_in_carousel_substrings=("puja", "pooja", "watti"),
        forbidden_substrings=_vcl._API_ERROR_FORBIDDEN,
    )
    clean = _carousel_html("Fruit Basket Deluxe", "Rose Bouquet")
    assert _evaluate_turn(scenario, clean) == []

    puja = _carousel_html("Puja Flower Set", "Fruit Basket")
    failures = _evaluate_turn(scenario, puja)
    assert any("forbidden substring in top carousel slot" in item for item in failures)


def test_delivery_followup_expects_delivery_markers() -> None:
    scenario = TurnScenario(
        name="delivery_followup",
        message="Can you deliver this to Colombo tomorrow?",
        expect_carousel=False,
        expect_delivery=True,
        forbidden_substrings=("Field required",),
    )
    html = "<p>Delivery to Colombo tomorrow is available.</p>"
    assert _evaluate_turn(scenario, html) == []


def test_roses_galle_tomorrow_rejects_unknown_city() -> None:
    scenario = TurnScenario(
        name="roses_galle_tomorrow",
        message="roses for Galle tomorrow",
        expect_carousel=False,
        expect_delivery=True,
        forbidden_substrings=_vcl._API_ERROR_FORBIDDEN,
    )
    html = "<p>Unknown city: Galle</p>"
    failures = _evaluate_turn(scenario, html)
    assert any("Unknown city" in item for item in failures)


def test_customer_eval_scenarios_registered() -> None:
    names = {scenario.name for scenario in _vcl.SCENARIOS}
    expected = {
        "cake_mom_colombo",
        "roses_galle_tomorrow",
        "gift_ideas_5000",
        "delivery_followup",
        "flowers_fruit_kandy",
        "track_vimp_regression",
    }
    assert expected <= names


def test_extract_top_carousel_card_texts_limits_slots() -> None:
    html = _carousel_html("Rose Bouquet", "Puja Set", "Fruit Basket", "Cake")
    texts = _extract_top_carousel_card_texts(html, limit=2)
    assert len(texts) == 2
    assert "rose bouquet" in texts[0]
    assert "puja set" in texts[1]
