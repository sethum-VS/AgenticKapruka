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


@pytest.fixture(autouse=True)
def cleanup_genai_patchers() -> None:
    """Clean up any active genai patchers created by test helpers."""
    yield
    try:
        from tests.helpers.mock_genai import ACTIVE_PATCHERS

        from lib.genai.completions import set_override_generate_content

        ACTIVE_PATCHERS.clear()
        set_override_generate_content(None)
    except ImportError:
        pass
