"""Unit tests for lib.chat.product_curation budget sorting and filtering."""

from __future__ import annotations

from lib.chat.product_curation import product_price_amount, sort_and_filter_by_budget


def _product(product_id: str, amount: float) -> dict[str, object]:
    return {
        "id": product_id,
        "name": f"Product {product_id}",
        "price": {"amount": amount, "currency": "LKR"},
        "in_stock": True,
    }


def test_sort_and_filter_in_budget_ascending() -> None:
    products = [
        _product("b", 7500.0),
        _product("a", 5000.0),
        _product("c", 7999.0),
    ]
    curated = sort_and_filter_by_budget(products, 8000.0, "LKR")
    assert [item["id"] for item in curated] == ["a", "b", "c"]


def test_sort_and_filter_hides_above_double_budget() -> None:
    products = [
        _product("cheap", 4000.0),
        _product("hidden", 17000.0),
        _product("edge", 16000.0),
    ]
    curated = sort_and_filter_by_budget(products, 8000.0, "LKR")
    ids = {item["id"] for item in curated}
    assert ids == {"cheap", "edge"}
    assert "hidden" not in ids


def test_sort_and_filter_near_budget_badge() -> None:
    products = [
        _product("in", 7000.0),
        _product("near", 8500.0),
        _product("far", 12000.0),
    ]
    curated = sort_and_filter_by_budget(products, 8000.0, "LKR")
    assert curated[0]["id"] == "in"
    assert curated[1]["id"] == "near"
    assert curated[1].get("slightly_over_budget") is True
    assert curated[2]["id"] == "far"
    assert "slightly_over_budget" not in curated[2]


def test_sort_and_filter_empty_budget_passthrough() -> None:
    products = [_product("x", 12000.0), _product("y", 3000.0)]
    assert sort_and_filter_by_budget(products, None, "LKR") == products
    assert sort_and_filter_by_budget(products, 0, "LKR") == products


def test_product_price_amount_reads_nested_price() -> None:
    assert product_price_amount(_product("p", 1234.0)) == 1234.0
    assert product_price_amount({"id": "p", "price": 99.0}) == 99.0
    assert product_price_amount({"id": "p"}) is None
