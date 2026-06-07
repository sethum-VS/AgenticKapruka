"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

APP_CSS = Path("static/css/app.css")
_MINIMAL_CSS = "/* pytest fixture */\n"


@pytest.fixture(autouse=True, scope="session")
def ensure_compiled_css() -> None:
    """Ensure generated app.css exists so static mount tests pass in CI."""
    APP_CSS.parent.mkdir(parents=True, exist_ok=True)
    if not APP_CSS.exists():
        APP_CSS.write_text(_MINIMAL_CSS, encoding="utf-8")
