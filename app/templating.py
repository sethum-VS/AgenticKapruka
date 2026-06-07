"""Jinja2 template environment for server-rendered HTMX pages."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import jinja2
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def format_currency(amount: float | int, currency: str = "LKR") -> str:
    """Stub currency filter; full formatting lands in PRD-056."""
    return f"{currency} {amount:,}"


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
    return Jinja2Templates(env=env)


def get_templates() -> Jinja2Templates:
    """FastAPI dependency returning the shared Jinja2 template environment."""
    return _create_templates()
