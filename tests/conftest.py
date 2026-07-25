"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_DEFAULT_TEST_ENV: dict[str, str] = {
    "REDIS_URL": "redis://localhost:6379/0",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "test-password",
    "ZEP_API_KEY": "zep-test-key",
    "NVIDIA_API_KEY": "nvidia-test-key",
    "KAPRUKA_MCP_URL": "https://mcp.kapruka.com/mcp",
    "SESSION_SECRET": "x" * 32,
}
for _key, _value in _DEFAULT_TEST_ENV.items():
    os.environ.setdefault(_key, _value)

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
