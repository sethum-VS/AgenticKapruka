"""Tests for pydantic-settings configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

_VALID_ENV: dict[str, str] = {
    "REDIS_URL": "redis://localhost:6379/0",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "test-password",
    "ZEP_API_KEY": "zep-test-key",
    "NVIDIA_API_KEY": "nvidia-test-key",
    "KAPRUKA_MCP_URL": "https://mcp.kapruka.com/mcp",
    "SESSION_SECRET": "x" * 32,
}


def _apply_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """All settings fields parse correctly from environment variables."""
    get_settings.cache_clear()
    _apply_env(monkeypatch, _VALID_ENV)

    settings = get_settings()

    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "test-password"
    assert settings.zep_api_key == "zep-test-key"
    assert settings.nvidia_api_key == "nvidia-test-key"
    assert settings.kapruka_mcp_url == "https://mcp.kapruka.com/mcp"
    assert settings.session_secret == "x" * 32
    assert settings.reranker_threshold == 0.45
    assert get_settings() is settings


def test_settings_nvidia_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    _apply_env(
        monkeypatch,
        {**_VALID_ENV, "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1"},
    )

    settings = get_settings()

    assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"


def test_settings_reranker_threshold_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    _apply_env(monkeypatch, {**_VALID_ENV, "RERANKER_THRESHOLD": "0.55"})

    settings = get_settings()

    assert settings.reranker_threshold == 0.55


def test_settings_rejects_short_session_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    env = {**_VALID_ENV, "SESSION_SECRET": "too-short"}
    _apply_env(monkeypatch, env)

    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_empty_production_key(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    env = {**_VALID_ENV, "ZEP_API_KEY": "   "}
    _apply_env(monkeypatch, env)

    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_empty_nvidia_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    env = {**_VALID_ENV, "NVIDIA_API_KEY": "   "}
    _apply_env(monkeypatch, env)

    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_whitespace_only_session_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_settings.cache_clear()
    env = {**_VALID_ENV, "SESSION_SECRET": " " * 32}
    _apply_env(monkeypatch, env)

    with pytest.raises(ValidationError):
        Settings()
