"""Unit tests for lib.chat.model_router and graphs.model_router."""

from __future__ import annotations

import pytest

from app.config import Settings
from graphs.state import AgentState
from lib.chat.model_router import (
    select_intent_model,
    select_rewrite_model,
    select_specialized_model,
)


def _state(**overrides: object) -> AgentState:
    base: AgentState = {
        "messages": [],
        "session_id": "sess-router-001",
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _nim_settings(**overrides: object) -> Settings:
    base = {
        "redis_url": "redis://localhost:6379/0",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "secret",
        "zep_api_key": "zep-key",
        "nvidia_api_key": "nvidia-key",
        "session_secret": "x" * 32,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_select_specialized_model_returns_nvidia_llm() -> None:
    settings = _nim_settings()
    assert select_specialized_model(settings=settings) == "z-ai/glm-5.2"


def test_select_intent_model_returns_nvidia_llm() -> None:
    settings = _nim_settings()
    assert select_intent_model(settings=settings) == "z-ai/glm-5.2"


def test_select_rewrite_model_returns_nvidia_llm() -> None:
    settings = _nim_settings()
    assert select_rewrite_model(settings=settings) == "z-ai/glm-5.2"


def test_select_specialized_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("ZEP_API_KEY", "zep-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setenv("SESSION_SECRET", "y" * 32)
    monkeypatch.setenv("NVIDIA_LLM_MODEL", "meta/llama3-70b-instruct")

    assert select_specialized_model() == "meta/llama3-70b-instruct"
