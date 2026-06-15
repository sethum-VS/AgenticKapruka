"""Unit tests for search broadening ladder on empty MCP results."""

from __future__ import annotations

from lib.chat.search_broadening import (
    BROADEN_LADDER,
    apply_first_broaden,
    broaden_search_args,
    build_empty_search_reply,
    first_applicable_broaden_step,
)


def test_broaden_gift_voucher_fallback_rewrites_gift_query() -> None:
    args = {"q": "gift ideas under 5000", "currency": "LKR", "max_price": 5000.0}
    broadened = broaden_search_args(args, "gift_voucher_fallback")
    assert broadened is not None
    assert broadened["q"] == "voucher"
    assert broadened["max_price"] == 5000.0


def test_broaden_gift_voucher_fallback_noop_without_gift() -> None:
    args = {"q": "birthday cake", "currency": "LKR"}
    assert broaden_search_args(args, "gift_voucher_fallback") is None


def test_broaden_gift_voucher_fallback_noop_when_already_voucher() -> None:
    args = {"q": "voucher", "currency": "LKR", "max_price": 5000.0}
    assert broaden_search_args(args, "gift_voucher_fallback") is None


def test_first_applicable_broaden_step_gift_query_prefers_voucher_fallback() -> None:
    args = {"q": "gift ideas", "currency": "LKR", "max_price": 5000.0}
    assert first_applicable_broaden_step(args) == "gift_voucher_fallback"


def test_broaden_simplify_q_birthday_cake_to_cake() -> None:
    args = {"q": "birthday cake for mom", "currency": "LKR", "max_price": 30.0}
    broadened = broaden_search_args(args, "simplify_q")
    assert broadened is not None
    assert broadened["q"] == "cake for mom"
    assert broadened["max_price"] == 30.0


def test_broaden_strip_city_removes_in_kandy() -> None:
    args = {"q": "cake for mom in Kandy", "currency": "LKR"}
    broadened = broaden_search_args(args, "strip_city")
    assert broadened is not None
    assert broadened["q"] == "cake for mom"


def test_broaden_drop_max_price() -> None:
    args = {"q": "cake", "currency": "LKR", "max_price": 30.0, "sort": "price_asc"}
    broadened = broaden_search_args(args, "drop_max_price")
    assert broadened is not None
    assert "max_price" not in broadened
    assert broadened["q"] == "cake"


def test_broaden_step_noop_when_not_applicable() -> None:
    args = {"q": "cake", "currency": "LKR"}
    assert broaden_search_args(args, "simplify_q") is None
    assert broaden_search_args(args, "strip_city") is None
    assert broaden_search_args(args, "drop_max_price") is None


def test_first_applicable_broaden_step_follows_ladder_order() -> None:
    args = {
        "q": "birthday cake for mom in Kandy",
        "currency": "LKR",
        "max_price": 30.0,
    }
    assert first_applicable_broaden_step(args) == "simplify_q"

    simplified = {"q": "cake for mom in Kandy", "currency": "LKR", "max_price": 30.0}
    assert first_applicable_broaden_step(simplified) == "strip_city"

    stripped = {"q": "cake for mom", "currency": "LKR", "max_price": 30.0}
    assert first_applicable_broaden_step(stripped) == "drop_max_price"

    minimal = {"q": "cake for mom", "currency": "LKR"}
    assert first_applicable_broaden_step(minimal) is None


def test_apply_first_broaden_returns_one_step() -> None:
    args = {
        "q": "birthday cake for mom in Kandy under $30",
        "currency": "LKR",
        "max_price": 30.0,
    }
    broadened, step = apply_first_broaden(args)
    assert step == "simplify_q"
    assert broadened is not None
    assert broadened["q"] == "cake for mom in Kandy under $30"


def test_broaden_ladder_order_constant() -> None:
    assert BROADEN_LADDER == (
        "gift_voucher_fallback",
        "simplify_q",
        "strip_city",
        "drop_max_price",
    )


def test_build_empty_search_reply_suggests_broader_query() -> None:
    base = build_empty_search_reply(broaden_attempted=False)
    assert "broader gift type" in base.lower()
    assert "budget" in base.lower()

    after_broaden = build_empty_search_reply(broaden_attempted=True)
    assert "broadened the search" in after_broaden.lower()
    assert "higher budget" in after_broaden.lower()
