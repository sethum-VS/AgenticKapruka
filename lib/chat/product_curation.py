"""Budget-aware sorting and filtering for product carousels."""

from __future__ import annotations

import re
from typing import Any

_NEAR_BUDGET_FACTOR = 1.10
_HIDE_BUDGET_FACTOR = 2.0

_FLOWER_FRUIT_INTENT = re.compile(
    r"\b(?:flower|flowers|rose|roses|bouquet|floral|fruit|fruits)\b",
    re.I,
)
_PUJA_DENYLIST = re.compile(r"\b(?:puja|pooja|pooj?a|watti|religious)\b", re.I)
_PRODUCE_DENYLIST = re.compile(
    r"\b(?:coconut|banana|grocery|vegetable|potato|onion|tomato|carrot)\b",
    re.I,
)
_BIRTHDAY_PRODUCT_RE = re.compile(r"\bbirthday\b", re.I)
_CAKE_ID_PREFIX = re.compile(r"^cake", re.I)
_DESSERT_CATEGORY_RE = re.compile(r"\b(?:chocolate|desserts?)\b", re.I)
_GENERIC_DESSERT_RE = re.compile(
    r"\b(?:lava\s+cake|dessert|mousse|brownie|loaf\s+cake|pudding|tiramisu)\b",
    re.I,
)
_CAKE_ACCESSORY_BLACKLIST = re.compile(
    r"\b(?:topper|mould|mold|turning\s+table|cake\s+stand|stand|icing\s+set|"
    r"fondant|nozzle|decorating|piping\s+bag|spatula)\b",
    re.I,
)
_FOCUS_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "chocolate": re.compile(r"\b(?:chocolate|choco|cocoa)\b", re.I),
    "cake": re.compile(r"\b(?:cake|birthday)\b", re.I),
    "flowers": re.compile(r"\b(?:flower|flowers|rose|roses|bouquet|floral)\b", re.I),
    "gift": re.compile(r"\b(?:hamper|combo|combopack|gift)\b", re.I),
}
_ANNIVERSARY_OCCASION_RE = re.compile(r"\banniversary\b", re.I)
_ANNIVERSARY_PROMOTE_RE = re.compile(
    r"\b(?:flower|flowers|rose|roses|bouquet|floral|cake|cakes|hamper|hampers|"
    r"chocolate|chocolates|combo|combopack)\b",
    re.I,
)
_ANNIVERSARY_DEMOTE_RE = re.compile(
    r"\b(?:greeting\s+card|watch\s+box|storage\s+box|voucher|gift\s+voucher)\b",
    re.I,
)

PUJA_NEGATIVE_CATEGORY_HINTS: tuple[str, ...] = (
    "Puja",
    "Pooja",
    "Religious offerings",
)


def is_flower_fruit_intent(query: str) -> bool:
    """True when the customer turn targets flowers, bouquets, or fruit gifts."""
    return bool(query.strip() and _FLOWER_FRUIT_INTENT.search(query))


def has_graph_hybrid_context(hybrid_context: dict[str, Any] | None) -> bool:
    """True when Neo4j GraphRAG fields are present in hybrid_context."""
    if not hybrid_context:
        return False
    return bool(
        hybrid_context.get("vector_hits")
        or hybrid_context.get("categories")
        or hybrid_context.get("occasions"),
    )


def product_price_amount(product: dict[str, Any]) -> float | None:
    """Return numeric price amount from a Kapruka search product dict."""
    raw_price = product.get("price")
    if isinstance(raw_price, dict):
        amount = raw_price.get("amount")
        if isinstance(amount, (int, float)):
            return float(amount)
        return None
    if isinstance(raw_price, (int, float)):
        return float(raw_price)
    return None


def _product_text_blob(product: dict[str, Any]) -> str:
    parts = [
        str(product.get("name") or ""),
        str(product.get("summary") or ""),
    ]
    category = product.get("category")
    if isinstance(category, dict):
        parts.extend(str(category.get(key) or "") for key in ("name", "slug", "id"))
    return " ".join(parts)


def product_matches_puja_denylist(product: dict[str, Any]) -> bool:
    """True when product name, summary, or category matches puja/religious denylist."""
    return bool(_PUJA_DENYLIST.search(_product_text_blob(product)))


def product_matches_produce_denylist(product: dict[str, Any]) -> bool:
    """True when product looks like grocery produce rather than a gift."""
    return bool(_PRODUCE_DENYLIST.search(_product_text_blob(product)))


def demote_produce_for_vague_gifts(
    products: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Light demotion of obvious produce when the query is a vague gift idea."""
    from lib.chat.intent_heuristics import is_vague_gift_intent

    if not is_vague_gift_intent(query):
        return products
    preferred = [product for product in products if not product_matches_produce_denylist(product)]
    demoted = [product for product in products if product_matches_produce_denylist(product)]
    return preferred + demoted


def _product_category_text(product: dict[str, Any]) -> str:
    category = product.get("category")
    if isinstance(category, dict):
        parts = [str(category.get(key) or "") for key in ("name", "slug", "id")]
        return " ".join(parts)
    return ""


def product_is_birthday_cake_product(product: dict[str, Any]) -> bool:
    """True when Kapruka metadata marks the item as a birthday cake."""
    if _BIRTHDAY_PRODUCT_RE.search(_product_category_text(product)):
        return True
    name = str(product.get("name") or "")
    if _BIRTHDAY_PRODUCT_RE.search(name):
        return True
    product_id = str(product.get("id") or "")
    return bool(_CAKE_ID_PREFIX.match(product_id) and _BIRTHDAY_PRODUCT_RE.search(name))


def product_is_generic_dessert(product: dict[str, Any]) -> bool:
    """True for chocolate/dessert items that are not birthday cakes."""
    if product_is_birthday_cake_product(product):
        return False
    if _DESSERT_CATEGORY_RE.search(_product_category_text(product)):
        return True
    return bool(_GENERIC_DESSERT_RE.search(_product_text_blob(product)))


def is_cake_accessory(product: dict[str, Any]) -> bool:
    """True when a search hit is a cake decorating tool rather than an edible cake."""
    return bool(_CAKE_ACCESSORY_BLACKLIST.search(_product_text_blob(product)))


def filter_cake_accessories(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop baking accessories and decorating tools from cake search results."""
    return [product for product in products if not is_cake_accessory(product)]


def apply_birthday_cake_curation(
    products: list[dict[str, Any]],
    *,
    query: str,
    hybrid_context: dict[str, Any] | None = None,
    graph_context_available: bool = False,
    session_product_focus: str | None = None,
) -> list[dict[str, Any]]:
    """Prefer birthday-category cakes and demote generic desserts for birthday turns."""
    from lib.neo4j.hybrid_context import is_birthday_cake_scoped_turn

    if not is_birthday_cake_scoped_turn(
        query,
        hybrid_context,
        session_product_focus=session_product_focus,
    ):
        return list(products)

    birthday: list[dict[str, Any]] = []
    neutral: list[dict[str, Any]] = []
    desserts: list[dict[str, Any]] = []
    for product in products:
        if product_is_birthday_cake_product(product):
            birthday.append(product)
        elif product_is_generic_dessert(product):
            desserts.append(product)
        else:
            neutral.append(product)

    if birthday:
        ordered = birthday + neutral
        if graph_context_available:
            return ordered + desserts
        return ordered

    if graph_context_available:
        return neutral + desserts
    return neutral


def demote_puja_products(
    products: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Move puja-denylist items to the end for flower/fruit discovery queries."""
    if not is_flower_fruit_intent(query):
        return list(products)
    preferred: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for product in products:
        if product_matches_puja_denylist(product):
            demoted.append(product)
        else:
            preferred.append(product)
    return preferred + demoted


def filter_puja_products(
    products: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Drop puja-denylist items for flower/fruit queries when GraphRAG is unavailable."""
    if not is_flower_fruit_intent(query):
        return list(products)
    return [product for product in products if not product_matches_puja_denylist(product)]


def apply_puja_curation(
    products: list[dict[str, Any]],
    *,
    query: str,
    graph_context_available: bool,
) -> list[dict[str, Any]]:
    """Demote puja items when graph hints exist; filter them when Neo4j is degraded."""
    if graph_context_available:
        return demote_puja_products(products, query)
    return filter_puja_products(products, query)


def _product_currency(product: dict[str, Any]) -> str:
    raw_price = product.get("price")
    if isinstance(raw_price, dict):
        code = raw_price.get("currency")
        if isinstance(code, str) and code.strip():
            return code.strip().upper()
    return "LKR"


def product_matches_focus(
    product: dict[str, Any],
    session_product_focus: str | None,
) -> bool:
    """True when product name/category matches the session shopping focus."""
    if not session_product_focus:
        return True
    pattern = _FOCUS_TOKEN_PATTERNS.get(session_product_focus)
    if pattern is None:
        return True
    return bool(pattern.search(_product_text_blob(product)))


def carousel_focus_guard(
    products: list[dict[str, Any]],
    session_product_focus: str | None,
    *,
    top_n: int = 5,
    min_ratio: float = 0.3,
) -> bool:
    """True when top carousel items align with session product focus."""
    if not products or not session_product_focus:
        return True
    sample = products[:top_n]
    if not sample:
        return True
    matches = sum(1 for product in sample if product_matches_focus(product, session_product_focus))
    return (matches / len(sample)) >= min_ratio


def refine_last_search_by_budget(
    last_search_products: list[dict[str, Any]],
    *,
    budget_max: float | None,
    currency: str,
    session_product_focus: str | None = None,
    session_search_query: str | None = None,
) -> list[dict[str, Any]] | None:
    """Re-filter prior carousel by budget and session focus; None triggers MCP fallback."""
    _ = session_search_query
    if not last_search_products or budget_max is None or budget_max <= 0:
        return None

    curated = sort_and_filter_by_budget(last_search_products, budget_max, currency)
    in_budget = [
        product
        for product in curated
        if (price := product_price_amount(product)) is not None and price <= budget_max
    ]
    if not in_budget:
        return None

    if session_product_focus:
        matching = [
            product
            for product in in_budget
            if product_matches_focus(product, session_product_focus)
        ]
        if not matching:
            return None
        demoted = [
            product
            for product in in_budget
            if not product_matches_focus(product, session_product_focus)
        ]
        return matching + demoted

    return in_budget


def is_anniversary_occasion_intent(
    query: str,
    hybrid_context: dict[str, Any] | None = None,
) -> bool:
    """True when the turn targets anniversary gifts."""
    if query.strip() and _ANNIVERSARY_OCCASION_RE.search(query):
        return True
    hints = (hybrid_context or {}).get("hints") or {}
    occasion = str(hints.get("occasion") or "").strip().lower()
    return occasion == "anniversary"


def apply_anniversary_curation(
    products: list[dict[str, Any]],
    *,
    query: str,
    hybrid_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Promote flowers/cakes/hampers and demote greeting cards for anniversary turns."""
    if not is_anniversary_occasion_intent(query, hybrid_context):
        return list(products)

    promoted: list[dict[str, Any]] = []
    neutral: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for product in products:
        blob = _product_text_blob(product)
        if _ANNIVERSARY_DEMOTE_RE.search(blob):
            demoted.append(product)
        elif _ANNIVERSARY_PROMOTE_RE.search(blob):
            promoted.append(product)
        else:
            neutral.append(product)
    return promoted + neutral + demoted


def sort_and_filter_by_budget(
    products: list[dict[str, Any]],
    budget_max: float | None,
    currency: str,
) -> list[dict[str, Any]]:
    """Hide items above 2× budget; sort in-budget asc, then near-budget (+10%) with badge."""
    if budget_max is None or budget_max <= 0:
        return list(products)

    target_currency = currency.strip().upper() if currency.strip() else "LKR"
    scoped = [
        product
        for product in products
        if _product_currency(product) == target_currency
    ]
    if not scoped:
        scoped = list(products)

    in_budget: list[dict[str, Any]] = []
    near_budget: list[dict[str, Any]] = []
    over_near: list[dict[str, Any]] = []

    for product in scoped:
        price = product_price_amount(product)
        if price is None:
            over_near.append(product)
            continue
        if price > budget_max * _HIDE_BUDGET_FACTOR:
            continue
        if price <= budget_max:
            in_budget.append(product)
        elif price <= budget_max * _NEAR_BUDGET_FACTOR:
            tagged = dict(product)
            tagged["slightly_over_budget"] = True
            near_budget.append(tagged)
        else:
            tagged = dict(product)
            tagged["over_budget"] = True
            over_near.append(tagged)

    in_budget.sort(key=lambda item: product_price_amount(item) or 0.0)
    near_budget.sort(key=lambda item: product_price_amount(item) or 0.0)
    over_near.sort(key=lambda item: product_price_amount(item) or 0.0)
    return in_budget + near_budget + over_near


def curate_carousel_products(
    products: list[dict[str, Any]],
    *,
    query: str,
    budget_max: float | None,
    currency: str,
    graph_context_available: bool = False,
    hybrid_context: dict[str, Any] | None = None,
    session_product_focus: str | None = None,
) -> list[dict[str, Any]]:
    """Apply birthday/puja relevance curation then budget-aware carousel ordering."""
    from lib.neo4j.hybrid_context import is_birthday_cake_scoped_turn

    scoped = apply_birthday_cake_curation(
        products,
        query=query,
        hybrid_context=hybrid_context,
        graph_context_available=graph_context_available,
        session_product_focus=session_product_focus,
    )
    scoped = apply_anniversary_curation(
        scoped,
        query=query,
        hybrid_context=hybrid_context,
    )
    scoped = demote_produce_for_vague_gifts(scoped, query)
    if is_birthday_cake_scoped_turn(
        query,
        hybrid_context,
        session_product_focus=session_product_focus,
    ):
        birthday_items = [
            product for product in scoped if product_is_birthday_cake_product(product)
        ]
        other_items = [
            product for product in scoped if not product_is_birthday_cake_product(product)
        ]
        return sort_and_filter_by_budget(
            birthday_items,
            budget_max,
            currency,
        ) + sort_and_filter_by_budget(other_items, budget_max, currency)
    if is_flower_fruit_intent(query):
        if graph_context_available:
            preferred = [
                product for product in scoped if not product_matches_puja_denylist(product)
            ]
            demoted = [product for product in scoped if product_matches_puja_denylist(product)]
            return sort_and_filter_by_budget(
                preferred,
                budget_max,
                currency,
            ) + sort_and_filter_by_budget(demoted, budget_max, currency)

        filtered = filter_puja_products(scoped, query)
        return sort_and_filter_by_budget(filtered, budget_max, currency)

    return sort_and_filter_by_budget(scoped, budget_max, currency)
