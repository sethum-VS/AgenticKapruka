"""Budget-aware sorting and filtering for product carousels."""

from __future__ import annotations

from typing import Any

_NEAR_BUDGET_FACTOR = 1.10
_HIDE_BUDGET_FACTOR = 2.0


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
