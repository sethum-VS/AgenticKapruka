"""Unit tests for lib.chat.product_curation budget sorting and filtering."""

from __future__ import annotations

from lib.chat.product_curation import (
    apply_birthday_cake_curation,
    apply_puja_curation,
    curate_carousel_products,
    demote_puja_products,
    filter_puja_products,
    has_graph_hybrid_context,
    is_flower_fruit_intent,
    product_is_birthday_cake_product,
    product_is_generic_dessert,
    product_matches_puja_denylist,
    product_price_amount,
    sort_and_filter_by_budget,
)


def _product(product_id: str, amount: float, *, name: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": product_id,
        "name": name or f"Product {product_id}",
        "price": {"amount": amount, "currency": "LKR"},
        "in_stock": True,
    }
    return payload


def _birthday_cake(product_id: str, amount: float, *, name: str | None = None) -> dict[str, object]:
    return {
        **_product(product_id, amount, name=name or "Chocolate Birthday Cake"),
        "category": {"id": "cat_birthday", "name": "Birthday", "slug": "birthday"},
    }


def _dessert(product_id: str, amount: float, *, name: str | None = None) -> dict[str, object]:
    return {
        **_product(product_id, amount, name=name or "Chocolate Lava Cake"),
        "category": {"id": "cat_dessert", "name": "Desserts", "slug": "desserts"},
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


def test_is_flower_fruit_intent_detects_flowers_and_fruit() -> None:
    assert is_flower_fruit_intent("flowers and fruit basket for Kandy")
    assert is_flower_fruit_intent("fresh roses bouquet")
    assert not is_flower_fruit_intent("birthday cake for mom")


def test_product_matches_puja_denylist_keywords() -> None:
    assert product_matches_puja_denylist({"name": "Puja Flower Set"})
    assert product_matches_puja_denylist({"name": "Watti Mal Pooja"})
    assert not product_matches_puja_denylist({"name": "Fruit Basket Deluxe"})


def test_filter_puja_products_removes_when_graph_down() -> None:
    products = [
        _product("fruit", 4000.0, name="Fruit Basket"),
        _product("puja", 3500.0, name="Puja Flower Set"),
    ]
    filtered = filter_puja_products(products, "flowers and fruit for Kandy")
    assert [item["id"] for item in filtered] == ["fruit"]


def test_demote_puja_products_moves_to_end_when_graph_up() -> None:
    products = [
        _product("puja", 3500.0, name="Puja Flower Set"),
        _product("fruit", 4000.0, name="Fruit Basket"),
    ]
    demoted = demote_puja_products(products, "flowers and fruit for Kandy")
    assert [item["id"] for item in demoted] == ["fruit", "puja"]


def test_apply_puja_curation_filters_or_demotes_by_graph_flag() -> None:
    products = [
        _product("puja", 3500.0, name="Puja Flower Set"),
        _product("fruit", 4000.0, name="Fruit Basket"),
    ]
    filtered = apply_puja_curation(
        products,
        query="flowers and fruit",
        graph_context_available=False,
    )
    assert [item["id"] for item in filtered] == ["fruit"]
    demoted = apply_puja_curation(
        products,
        query="flowers and fruit",
        graph_context_available=True,
    )
    assert [item["id"] for item in demoted] == ["fruit", "puja"]


def test_has_graph_hybrid_context_detects_neo4j_fields() -> None:
    assert has_graph_hybrid_context({"vector_hits": [{"id": "category:flowers"}]})
    assert not has_graph_hybrid_context({"hints": {"category": "Flowers"}})
    assert not has_graph_hybrid_context(None)


def test_curate_carousel_products_puja_and_budget_order() -> None:
    products = [
        _product("puja", 3500.0, name="Puja Flower Set"),
        _product("over", 6500.0, name="Premium Fruit Hamper"),
        _product("in", 4500.0, name="Rose Bouquet"),
    ]
    curated = curate_carousel_products(
        products,
        query="flowers and fruit basket budget 5000",
        budget_max=5000.0,
        currency="LKR",
        graph_context_available=False,
    )
    assert [item["id"] for item in curated] == ["in", "over"]
    assert "puja" not in {item["id"] for item in curated}


def test_product_is_birthday_cake_product_detects_category_and_name() -> None:
    assert product_is_birthday_cake_product(_birthday_cake("cake01", 2500.0))
    assert not product_is_generic_dessert(_birthday_cake("cake01", 2500.0))
    assert product_is_generic_dessert(_dessert("dess01", 750.0))


def test_apply_birthday_cake_curation_prefers_birthday_over_desserts() -> None:
    products = [
        _dessert("lava", 890.0),
        _birthday_cake("bday", 2550.0),
        _dessert("loaf", 750.0, name="Chocolate Loaf Cake"),
    ]
    curated = apply_birthday_cake_curation(
        products,
        query="chocolate birthday cake for wife budget 3000 Kandy",
        graph_context_available=True,
    )
    assert curated[0]["id"] == "bday"
    assert curated[-1]["id"] == "loaf"


def test_apply_birthday_cake_curation_filters_desserts_when_graph_down() -> None:
    products = [
        _dessert("lava", 890.0),
        _birthday_cake("bday", 2550.0),
    ]
    curated = apply_birthday_cake_curation(
        products,
        query="birthday cake for mom",
        graph_context_available=False,
    )
    assert [item["id"] for item in curated] == ["bday"]


def test_curate_carousel_products_birthday_cake_budget_order() -> None:
    products = [
        _dessert("lava", 890.0),
        _birthday_cake("bday", 2550.0),
        _birthday_cake("premium", 7800.0, name="Premium Birthday Cake"),
    ]
    curated = curate_carousel_products(
        products,
        query="chocolate birthday cake budget 3000",
        budget_max=3000.0,
        currency="LKR",
        graph_context_available=True,
        hybrid_context={"hints": {"occasion": "Birthday"}},
    )
    assert [item["id"] for item in curated] == ["bday", "lava"]
