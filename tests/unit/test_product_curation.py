"""Unit tests for lib.chat.product_curation budget sorting and filtering."""

from __future__ import annotations

from lib.chat.product_curation import (
    apply_anniversary_curation,
    apply_birthday_cake_curation,
    apply_gift_curation,
    apply_puja_curation,
    apply_recipient_curation,
    carousel_focus_guard,
    curate_carousel_products,
    demote_non_chocolate_for_chocolate_focus,
    demote_non_floral_for_flower_intent,
    demote_off_focus_products,
    ensure_flower_price_tier_diversity,
    demote_puja_products,
    enrich_carousel_product_descriptions,
    enrich_product_card_description,
    filter_gift_noise_products,
    filter_puja_products,
    has_graph_hybrid_context,
    is_flower_fruit_intent,
    product_is_birthday_cake_product,
    product_is_generic_dessert,
    product_matches_focus,
    product_matches_puja_denylist,
    product_price_amount,
    refine_last_search_by_budget,
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
    assert curated[2].get("over_budget") is True
    assert "slightly_over_budget" not in curated[2]


def test_sort_and_filter_over_near_budget_badge_for_5950_on_5000() -> None:
    products = [_product("gift", 5950.0)]
    curated = sort_and_filter_by_budget(products, 5000.0, "LKR")
    assert len(curated) == 1
    assert curated[0].get("over_budget") is True


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


def test_refine_last_search_by_budget_filters_chocolate_and_drops_over_budget() -> None:
    products = [
        _product("choc1", 5500.0, name="Cadbury Chocolate Gift Box"),
        _product("choc2", 7500.0, name="Premium Chocolate Hamper"),
        _product("card", 1200.0, name="Greeting Card"),
    ]
    refined = refine_last_search_by_budget(
        products,
        budget_max=6000.0,
        currency="LKR",
        session_product_focus="chocolate",
    )
    assert refined is not None
    assert all(
        float(item["price"]["amount"]) <= 6000.0  # type: ignore[index]
        for item in refined
        if isinstance(item.get("price"), dict)
    )
    assert refined[0]["id"] == "choc1"
    assert "choc2" not in {item["id"] for item in refined}
    assert "card" not in {item["id"] for item in refined}


def test_refine_last_search_by_budget_returns_none_without_focus_match() -> None:
    products = [
        _product("card", 1200.0, name="Greeting Card"),
        _product("voucher", 5000.0, name="Gift Voucher"),
    ]
    assert (
        refine_last_search_by_budget(
            products,
            budget_max=6000.0,
            currency="LKR",
            session_product_focus="chocolate",
        )
        is None
    )


def test_carousel_focus_guard_detects_off_topic_drift() -> None:
    greeting_cards = [
        _product(f"card{i}", 1000.0 + i, name=f"Greeting Card {i}") for i in range(5)
    ]
    assert not carousel_focus_guard(greeting_cards, "chocolate")
    mixed = [
        _product("choc1", 4500.0, name="Chocolate Truffles"),
        _product("choc2", 5200.0, name="Dark Chocolate Box"),
        *greeting_cards[:3],
    ]
    assert carousel_focus_guard(mixed, "chocolate")


def test_apply_anniversary_curation_demotes_greeting_cards() -> None:
    products = [
        _product("card", 1500.0, name="Anniversary Greeting Card"),
        _product("roses", 6500.0, name="Red Rose Bouquet"),
        _product("hamper", 8900.0, name="Anniversary Gift Hamper"),
    ]
    curated = apply_anniversary_curation(
        products,
        query="Show me some anniversary gifts",
        hybrid_context={"hints": {"occasion": "anniversary"}},
    )
    assert curated[0]["id"] in {"roses", "hamper"}
    assert curated[-1]["id"] == "card"


def test_product_matches_focus_chocolate_tokens() -> None:
    assert product_matches_focus(
        _product("x", 100.0, name="Dark Choco Truffles"),
        "chocolate",
    )
    assert not product_matches_focus(
        _product("x", 100.0, name="Greeting Card"),
        "chocolate",
    )


def test_demote_non_floral_for_flower_intent() -> None:
    products = [
        _product("freshener", 1200.0, name="Alco Fresh Rose Air Freshener"),
        _product("roses", 4500.0, name="Blush Roses Bouquet"),
    ]
    curated = demote_non_floral_for_flower_intent(products, "fresh roses bouquet")
    assert curated[0]["id"] == "roses"
    assert curated[-1]["id"] == "freshener"


def test_sort_and_filter_strict_budget_hides_over_cap() -> None:
    products = [
        _product("in", 4500.0),
        _product("over", 6500.0),
    ]
    curated = sort_and_filter_by_budget(products, 5000.0, "LKR", strict_in_budget=True)
    assert [item["id"] for item in curated] == ["in"]


def test_apply_birthday_cake_curation_boosts_chocolate_with_flavor_hint() -> None:
    products = [
        _product("vanilla", 3500.0, name="Vanilla Birthday Cake"),
        _product("choc", 3800.0, name="Chocolate Birthday Cake"),
    ]
    curated = apply_birthday_cake_curation(
        products,
        query="chocolate birthday cake for mom",
        hybrid_context={"hints": {"occasion": "Birthday"}},
        session_product_focus="cake",
        session_flavor_hint="chocolate",
    )
    assert curated[0]["id"] == "choc"


def test_apply_gift_curation_demotes_vouchers_unless_requested() -> None:
    products = [
        _product("voucher", 4500.0, name="Kapruka Gift Voucher"),
        _product("hamper", 4800.0, name="Birthday Gift Hamper"),
    ]
    curated = apply_gift_curation(
        products,
        session_product_focus="gift",
        user_message="Gift ideas under Rs. 5,000",
    )
    assert curated[0]["id"] == "hamper"
    assert curated[-1]["id"] == "voucher"


def test_ensure_flower_price_tier_diversity_promotes_affordable_rose() -> None:
    products = [
        _product("premium", 4800.0, name="Premium Rose Bouquet"),
        _product("deluxe", 4600.0, name="Deluxe Roses"),
        _product("cheap", 3200.0, name="6 Red Rose Bouquet"),
        _product("mid", 4200.0, name="Classic Roses"),
        _product("high", 4900.0, name="Grand Roses"),
    ]
    curated = ensure_flower_price_tier_diversity(products, 5000.0)
    top_ids = {item["id"] for item in curated[:3]}
    assert "cheap" in top_ids


def test_curate_carousel_products_roses_budget_skips_phrase_boost() -> None:
    products = [
        _product("cheap", 3200.0, name="6 Red Rose Bouquet"),
        _product("premium", 4800.0, name="Premium Rose Bouquet"),
    ]
    curated = curate_carousel_products(
        products,
        query="fresh roses under 5000 LKR",
        budget_max=5000.0,
        currency="LKR",
        strict_budget=True,
    )
    assert curated[0]["id"] == "cheap"


def test_apply_gift_curation_promotes_hampers_over_convenience_candy() -> None:
    products = [
        _product("curry", 800.0, name="Curry Powder Gift Pack"),
        _product("kitkat", 1200.0, name="KitKat Minis"),
        _product("ferrero", 4500.0, name="Ferrero Rocher Chocolate Gift Box"),
        _product("hamper", 5500.0, name="Birthday Chocolate Hamper"),
        _product("bouquet", 4800.0, name="Rose Chocolate Bouquet"),
    ]
    curated = apply_gift_curation(
        products,
        session_product_focus="chocolate",
        user_message="wife birthday chocolate under 6000",
    )
    top_ids = {item["id"] for item in curated[:2]}
    assert "bouquet" not in top_ids
    assert top_ids & {"hamper", "ferrero"}
    assert curated[-1]["id"] in {"curry", "kitkat"}


def test_demote_off_focus_products_keeps_matches_first() -> None:
    products = [
        _product("card", 1200.0, name="Greeting Card"),
        _product("choc", 4500.0, name="Chocolate Truffles"),
    ]
    demoted = demote_off_focus_products(products, "chocolate")
    assert demoted[0]["id"] == "choc"


def test_demote_non_chocolate_for_chocolate_focus_demotes_bouquets() -> None:
    products = [
        _product("roses", 4500.0, name="Blush Roses Bouquet"),
        _product("choc", 5200.0, name="Dark Chocolate Truffles"),
    ]
    curated = demote_non_chocolate_for_chocolate_focus(
        products,
        "wife birthday chocolate under 6000",
        session_product_focus="chocolate",
    )
    assert curated[0]["id"] == "choc"
    assert curated[-1]["id"] == "roses"


def test_apply_recipient_curation_drops_title_leading_dad_box_for_wife() -> None:
    products = [
        _product("dad", 4500.0, name="Dad Chocolate Gift Box"),
        _product("her1", 5200.0, name="Chocolate Truffles Gift Box"),
        _product("her2", 4000.0, name="Heart Chocolate Assortment"),
        _product("her3", 3500.0, name="Luxury Chocolate Hamper"),
    ]
    curated = apply_recipient_curation(products, "wife")
    ids = [item["id"] for item in curated]
    assert "dad" not in ids


def test_curate_carousel_products_wife_birthday_chocolate_excludes_bouquet_and_dad() -> None:
    products = [
        _product("bouquet", 4500.0, name="Rose Bouquet Premium"),
        _product("dad", 4200.0, name="Dad Chocolate Gift Box"),
        _product("choc1", 5200.0, name="Heart Chocolate Gift Box"),
        _product("choc2", 4800.0, name="Ferrero Rocher Collection"),
        _product("choc3", 3900.0, name="Milk Chocolate Truffles"),
    ]
    curated = curate_carousel_products(
        products,
        query="wife birthday chocolate under 6000",
        budget_max=6000.0,
        currency="LKR",
        session_product_focus="chocolate",
        session_recipient_hint="wife",
        strict_budget=True,
        hybrid_context={"hints": {"occasion": "Birthday"}},
    )
    names = [str(item.get("name") or "").lower() for item in curated]
    assert not any("bouquet" in name for name in names)
    assert not any(name.startswith("dad") for name in names)


def test_filter_gift_noise_products_drops_curry_and_snack_bar() -> None:
    products = [
        _product("curry", 350.0, name="Ruhunu Curry Powder 250g"),
        _product("snack", 70.0, name="Chocolate Snack Bar Mini"),
        _birthday_cake("bento", 3660.0, name="Celebratory Duo Bento"),
    ]
    filtered = filter_gift_noise_products(products, strict=True)
    assert [item["id"] for item in filtered] == ["bento"]


def test_apply_recipient_curation_demotes_for_him_on_wife_flow() -> None:
    products = [
        _product("him", 4500.0, name="Gentleman's Giftset For Him"),
        _product("her", 5200.0, name="Chocolate Truffles Gift Box"),
    ]
    curated = apply_recipient_curation(products, "wife")
    assert curated[0]["id"] == "her"
    assert curated[-1]["id"] == "him"


def test_curate_carousel_products_strict_budget_excludes_noise() -> None:
    products = [
        _product("curry", 350.0, name="Ruhunu Curry Powder"),
        _product("snack", 70.0, name="KitKat Snack Bar"),
        _birthday_cake("bento", 3660.0, name="Say Cheers Chocolate Cake"),
    ]
    curated = curate_carousel_products(
        products,
        query="wife birthday chocolate under 6000",
        budget_max=6000.0,
        currency="LKR",
        session_product_focus="chocolate",
        session_recipient_hint="wife",
        strict_budget=True,
        hybrid_context={"hints": {"occasion": "Birthday"}},
    )
    names = [str(item.get("name") or "") for item in curated]
    assert not any("curry" in name.lower() for name in names)
    assert not any("snack" in name.lower() or "kitkat" in name.lower() for name in names)
    assert any("cake" in name.lower() or "bento" in name.lower() for name in names)


# ── Phase 1c: Drop filter tests ──────────────────────────────────────────────


def test_apply_recipient_curation_drops_for_him_on_wife_when_enough_remain() -> None:
    """Wife flow: 'for dad' / 'father's gift' items are dropped when ≥3 items remain."""
    products = [
        _product("dad1", 4500.0, name="Gift for Dad Blue Heart"),
        _product("dad2", 3200.0, name="Father's Day Special Combo"),
        _product("her1", 5200.0, name="Chocolate Truffles Gift Box"),
        _product("her2", 4000.0, name="Rose Bouquet Premium"),
        _product("her3", 3500.0, name="Ladies Spa Gift Set"),
    ]
    curated = apply_recipient_curation(products, "wife")
    ids = [item["id"] for item in curated]
    assert "dad1" not in ids, "for dad should be dropped from wife flow"
    assert "dad2" not in ids, "father's should be dropped from wife flow"
    assert len(curated) >= 3


def test_apply_recipient_curation_falls_back_to_demote_when_few_remain() -> None:
    """Fall back to demote-only when dropping would leave fewer than 3 items."""
    products = [
        _product("dad1", 4500.0, name="Gift for Dad Combo"),
        _product("her1", 5200.0, name="Chocolate Gift Box"),
        _product("her2", 4000.0, name="Rose Bouquet"),
    ]
    curated = apply_recipient_curation(products, "wife")
    ids = [item["id"] for item in curated]
    # 2 preferred items — fallback: demoted item appended at end
    assert ids[-1] == "dad1", "mismatched item falls to end on demote fallback"
    assert len(curated) == 3


def test_apply_recipient_curation_drops_for_her_on_dad_flow() -> None:
    """Dad flow: 'for her' / 'ladies' items are dropped when ≥3 items remain."""
    products = [
        _product("her1", 4500.0, name="Ladies Perfume For Her"),
        _product("him1", 3200.0, name="Men's Grooming Kit"),
        _product("him2", 5000.0, name="Craft Beer Gift Set"),
        _product("him3", 4200.0, name="Gent's Watch Box"),
    ]
    curated = apply_recipient_curation(products, "dad")
    ids = [item["id"] for item in curated]
    assert "her1" not in ids, "for her should be dropped from dad flow"


def test_apply_anniversary_curation_drops_watch_box_when_enough_remain() -> None:
    """Anniversary: watch box / greeting card dropped when ≥3 items remain."""
    products = [
        _product("card1", 350.0, name="Greeting Card With Envelope"),
        _product("box1", 2800.0, name="Wooden Watch Storage Box"),
        _product("flower1", 4500.0, name="Red Roses Anniversary Bouquet"),
        _product("cake1", 3200.0, name="Anniversary Cake Heart"),
        _product("hamper1", 6500.0, name="Anniversary Gift Hamper"),
    ]
    curated = apply_anniversary_curation(
        products,
        query="anniversary gift for wife",
        hybrid_context=None,
    )
    ids = [item["id"] for item in curated]
    assert "card1" not in ids, "greeting card should be dropped"
    assert "box1" not in ids, "watch storage box should be dropped"
    assert len(curated) >= 3


def test_apply_anniversary_curation_session_occasion_triggers_without_word() -> None:
    """session_occasion='anniversary' activates curation even when word absent from query."""
    products = [
        _product("card1", 350.0, name="Greeting Card Premium"),
        _product("flower1", 4500.0, name="Red Roses Bouquet"),
        _product("cake1", 3200.0, name="Chocolate Celebration Cake"),
        _product("hamper1", 6500.0, name="Luxury Gift Hamper"),
    ]
    curated = apply_anniversary_curation(
        products,
        query="gift ideas for my wife",
        session_occasion="anniversary",
    )
    ids = [item["id"] for item in curated]
    assert "card1" not in ids, "greeting card should be dropped on session_occasion=anniversary"
    assert "flower1" in ids


def test_apply_anniversary_curation_fallback_demote_when_few_remain() -> None:
    """Fall back to demote when dropping would leave fewer than 3 items."""
    products = [
        _product("card1", 350.0, name="Greeting Card Birthday"),
        _product("flower1", 4500.0, name="Rose Bouquet"),
        _product("hamper1", 6500.0, name="Anniversary Hamper"),
    ]
    curated = apply_anniversary_curation(
        products,
        query="anniversary flowers",
    )
    assert len(curated) == 3, "all items present on fallback"
    assert curated[-1]["id"] == "card1", "greeting card demoted to end"


def test_enrich_product_card_description_uses_mcp_summary() -> None:
    product = {
        "id": "cake1",
        "name": "Springtime Birthday Ribbon Cake",
        "summary": "Delicate sponge with ribbon decoration and buttercream.",
        "price": {"amount": 5770.0, "currency": "LKR"},
    }
    enriched = enrich_product_card_description(product)
    assert enriched["card_description_fallback"] == (
        "Delicate sponge with ribbon decoration and buttercream."
    )


def test_enrich_product_card_description_truncates_long_summary() -> None:
    long_summary = " ".join(["word"] * 30)
    product = {"id": "cake1", "name": "Cake", "summary": long_summary}
    enriched = enrich_product_card_description(product)
    fallback = str(enriched["card_description_fallback"])
    assert len(fallback) <= 96
    assert fallback.endswith("…")


def test_enrich_carousel_product_descriptions_applies_to_all_items() -> None:
    products = [
        {"id": "a", "name": "Cake A", "summary": "Summary A."},
        {"id": "b", "name": "Cake B", "summary": "Summary B."},
    ]
    enriched = enrich_carousel_product_descriptions(products)
    assert enriched[0]["card_description_fallback"] == "Summary A."
    assert enriched[1]["card_description_fallback"] == "Summary B."
