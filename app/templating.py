"""Jinja2 template environment for server-rendered HTMX pages."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import jinja2
from fastapi.templating import Jinja2Templates

from lib.redis.cart import StoredCartItem
from lib.utils.currency import SUPPORTED_CURRENCIES, format_currency

SUPPORTED_CURRENCY_CODES: tuple[str, ...] = tuple(sorted(SUPPORTED_CURRENCIES))

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def urlencode_filter(value: str) -> str:
    """URL-encode query parameter values for HTMX hx-get URLs."""
    return quote(str(value), safe="")


def _is_dev_environment() -> bool:
    return os.getenv("APP_ENV", "development").lower() != "production"


@lru_cache
def _create_templates() -> Jinja2Templates:
    loader = jinja2.FileSystemLoader(str(TEMPLATES_DIR))
    env = jinja2.Environment(
        loader=loader,
        autoescape=jinja2.select_autoescape(),
        auto_reload=_is_dev_environment(),
    )
    env.filters["format_currency"] = format_currency
    env.filters["urlencode"] = urlencode_filter
    return Jinja2Templates(env=env)


def get_templates() -> Jinja2Templates:
    """FastAPI dependency returning the shared Jinja2 template environment."""
    return _create_templates()


def render_stock_badge(*, in_stock: bool, stock_level: str = "high") -> str:
    """Render templates/components/stock_badge.html for product image overlays."""
    templates = get_templates()
    template = templates.env.get_template("components/stock_badge.html")
    return template.render(in_stock=in_stock, stock_level=stock_level)


def render_product_card(product: dict[str, Any]) -> str:
    """Render templates/components/product_card.html for carousel and search results."""
    templates = get_templates()
    template = templates.env.get_template("components/product_card.html")
    return template.render(product=product)


def render_product_carousel(products: list[dict[str, Any]]) -> str:
    """Render templates/components/product_carousel.html with product_card partials."""
    templates = get_templates()
    template = templates.env.get_template("components/product_carousel.html")
    return template.render(products=products)


def categories_for_chips(hybrid_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract deduplicated category rows from hybrid_context for chip rendering."""
    if not hybrid_context:
        return []

    seen: set[str] = set()
    chips: list[dict[str, Any]] = []
    for source in ("vector_hits", "categories"):
        for item in hybrid_context.get(source) or []:
            name = item.get("display_name")
            if not name or name in seen:
                continue
            seen.add(str(name))
            chips.append(dict(item))
    return chips


def render_category_chips(
    categories: list[dict[str, Any]],
    *,
    active_category: str | None = None,
) -> str:
    """Render templates/components/category_chips.html for hybrid_context category filters."""
    templates = get_templates()
    template = templates.env.get_template("components/category_chips.html")
    return template.render(categories=categories, active_category=active_category)


def render_cart_partial(
    *,
    items: list[StoredCartItem],
    currency: str = "LKR",
) -> str:
    """Render templates/checkout/cart_partial.html for HTMX outerHTML swaps."""
    templates = get_templates()
    template = templates.env.get_template("checkout/cart_partial.html")
    return template.render(items=items, currency=currency)


def render_currency_selector(*, currency: str = "LKR") -> str:
    """Render templates/components/currency_selector.html for the site header."""
    templates = get_templates()
    template = templates.env.get_template("components/currency_selector.html")
    return template.render(
        currency=currency,
        supported_currencies=SUPPORTED_CURRENCY_CODES,
    )


def normalize_html_snapshot(html: str) -> str:
    """Collapse insignificant whitespace for stable snapshot comparisons."""
    collapsed = re.sub(r">\s+<", "><", html.strip())
    return re.sub(r"\s{2,}", " ", collapsed)
