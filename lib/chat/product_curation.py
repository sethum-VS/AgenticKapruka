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
_BIRTHDAY_PRODUCT_RE = re.compile(r"\bbirthday\b", re.I)
_CAKE_ID_PREFIX = re.compile(r"^cake", re.I)
_DESSERT_CATEGORY_RE = re.compile(r"\b(?:chocolate|desserts?)\b", re.I)
_GENERIC_DESSERT_RE = re.compile(
    r"\b(?:lava\s+cake|dessert|mousse|brownie|loaf\s+cake|pudding|tiramisu)\b",
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


def apply_birthday_cake_curation(
    products: list[dict[str, Any]],
    *,
    query: str,
    hybrid_context: dict[str, Any] | None = None,
    graph_context_available: bool = False,
) -> list[dict[str, Any]]:
    """Prefer birthday-category cakes and demote generic desserts for birthday turns."""
    from lib.neo4j.hybrid_context import is_birthday_cake_scoped_turn

    if not is_birthday_cake_scoped_turn(query, hybrid_context):
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


def sort_and_filter_by_budget(
    products: list[dict[str, Any]],
    budget_max: float | None,
    currency: str,
) -> list[dict[str, Any]]:
    """Hide items above 2× budget; sort in-budget asc, then near-budget (+10%) with badge.

    ``currency`` is accepted for API symmetry with session currency (prices are already
    normalized by Kapruka MCP for the requested currency).
    """
    _ = currency
    if budget_max is None or budget_max <= 0:
        return list(products)

    in_budget: list[dict[str, Any]] = []
    near_budget: list[dict[str, Any]] = []
    over_near: list[dict[str, Any]] = []

    for product in products:
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
            over_near.append(product)

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
) -> list[dict[str, Any]]:
    """Apply birthday/puja relevance curation then budget-aware carousel ordering."""
    from lib.neo4j.hybrid_context import is_birthday_cake_scoped_turn

    scoped = apply_birthday_cake_curation(
        products,
        query=query,
        hybrid_context=hybrid_context,
        graph_context_available=graph_context_available,
    )
    if is_birthday_cake_scoped_turn(query, hybrid_context):
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
