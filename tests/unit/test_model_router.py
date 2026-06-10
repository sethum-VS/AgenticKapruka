"""Unit tests for graphs.model_router and lib.chat.model_router."""

from __future__ import annotations

import pytest

from app.config import Settings
from graphs.model_router import FLASH_MODEL, PRO_MODEL, select_model, select_model_tier
from graphs.state import AgentState
from lib.chat.model_router import (
    build_lora_endpoint_resource,
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


def test_select_model_defaults_to_flash() -> None:
    assert select_model(_state()) == FLASH_MODEL
    assert select_model_tier(_state()) == "flash"


def test_select_model_escalates_on_checkout_review() -> None:
    state = _state(checkout_state="review")
    assert select_model_tier(state) == "pro"
    assert select_model(state) == PRO_MODEL


def test_select_model_respects_explicit_pro_tier() -> None:
    state = _state(model_tier="pro")
    assert select_model(state) == PRO_MODEL


def test_select_model_stays_flash_at_tool_call_threshold() -> None:
    state = _state(tool_call_count=3)
    assert select_model_tier(state) == "flash"
    assert select_model(state) == FLASH_MODEL


def test_select_model_escalates_after_fourth_tool_call() -> None:
    """tool_call_count > 3 triggers Pro escalation (4th call onward)."""
    state = _state(tool_call_count=4)
    assert select_model_tier(state) == "pro"
    assert select_model(state) == PRO_MODEL


def test_select_model_escalates_when_tool_call_count_exceeds_threshold() -> None:
    state = _state(tool_call_count=10)
    assert select_model(state) == PRO_MODEL


def _lora_settings(**overrides: object) -> Settings:
    base = {
        "redis_url": "redis://localhost:6379/0",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "secret",
        "zep_api_key": "zep-key",
        "gcp_project_id": "kapruka-project",
        "gcp_location": "us-central1",
        "session_secret": "x" * 32,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_build_lora_endpoint_resource() -> None:
    resource = build_lora_endpoint_resource(
        "kapruka-project",
        "us-central1",
        "1234567890",
    )
    assert resource == "projects/kapruka-project/locations/us-central1/endpoints/1234567890"


def test_select_specialized_model_defaults_to_flash_without_lora() -> None:
    settings = _lora_settings()
    assert select_specialized_model(settings=settings) == FLASH_MODEL
    assert select_intent_model(settings=settings) == FLASH_MODEL
    assert select_rewrite_model(settings=settings) == FLASH_MODEL


def test_select_specialized_model_uses_lora_endpoint_when_configured() -> None:
    settings = _lora_settings(kapruka_lora_endpoint_id="lora-endpoint-42")
    expected = "projects/kapruka-project/locations/us-central1/endpoints/lora-endpoint-42"
    assert select_specialized_model(settings=settings) == expected
    assert select_intent_model(settings=settings) == expected
    assert select_rewrite_model(settings=settings) == expected


def test_select_specialized_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("ZEP_API_KEY", "zep-key")
    monkeypatch.setenv("GCP_PROJECT_ID", "env-project")
    monkeypatch.setenv("GCP_LOCATION", "asia-south1")
    monkeypatch.setenv("SESSION_SECRET", "y" * 32)
    monkeypatch.setenv("KAPRUKA_LORA_ENDPOINT_ID", "env-lora-99")

    model = select_specialized_model()
    assert model == "projects/env-project/locations/asia-south1/endpoints/env-lora-99"
