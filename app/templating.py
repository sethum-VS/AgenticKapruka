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

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def format_currency(amount: float | int, currency: str = "LKR") -> str:
    """Stub currency filter; full formatting lands in PRD-056."""
    return f"{currency} {amount:,}"


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


def normalize_html_snapshot(html: str) -> str:
    """Collapse insignificant whitespace for stable snapshot comparisons."""
    collapsed = re.sub(r">\s+<", "><", html.strip())
    return re.sub(r"\s{2,}", " ", collapsed)
