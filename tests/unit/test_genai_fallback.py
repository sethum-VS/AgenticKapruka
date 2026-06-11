"""Tests for lib.genai.fallback multi-region Vertex failover."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from app.config import Settings
from lib.genai.fallback import (
    clear_client_cache,
    generate_content_with_fallback,
    vertex_location_chain,
)


def _vertex_settings(**overrides: object) -> Settings:
    base = {
        "redis_url": "redis://localhost:6379/0",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "test-password",
        "zep_api_key": "zep-test-key",
        "gcp_project_id": "test-project",
        "gcp_location": "us-central1",
        "session_secret": "x" * 32,
        "gemini_backend": "vertex",
        "gemini_chat_location": "global",
        "gemini_fallback_regions": ["europe-west4", "us-east4"],
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _clear_fallback_client_cache() -> None:
    clear_client_cache()


def test_vertex_location_chain_deduplicates_primary() -> None:
    settings = _vertex_settings(
        gemini_chat_location="us-central1",
        gemini_fallback_regions=["us-central1", "europe-west4"],
    )
    assert vertex_location_chain(settings) == ["us-central1", "europe-west4"]


def test_generate_content_with_fallback_uses_injected_client_only() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    config = types.GenerateContentConfig(temperature=0)

    result = generate_content_with_fallback(
        client=mock_client,
        model="gemini-2.5-flash",
        contents="hello",
        config=config,
    )

    assert result is mock_response
    mock_client.models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash",
        contents="hello",
        config=config,
    )


def test_generate_content_with_fallback_cascades_on_429() -> None:
    settings = _vertex_settings()
    config = types.GenerateContentConfig(temperature=0)
    primary_client = MagicMock()
    fallback_client = MagicMock()
    success_response = MagicMock()
    rate_limit = genai_errors.ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})
    primary_client.models.generate_content.side_effect = rate_limit
    fallback_client.models.generate_content.return_value = success_response

    with patch(
        "lib.genai.fallback.create_genai_client",
        side_effect=[primary_client, fallback_client],
    ):
        result = generate_content_with_fallback(
            settings=settings,
            model="gemini-2.5-flash",
            contents="hello",
            config=config,
        )

    assert result is success_response
    assert primary_client.models.generate_content.call_count == 1
    assert fallback_client.models.generate_content.call_count == 1


def test_generate_content_with_fallback_raises_after_all_regions_exhausted() -> None:
    settings = _vertex_settings(gemini_fallback_regions=[])
    config = types.GenerateContentConfig(temperature=0)
    exhausted_client = MagicMock()
    rate_limit = genai_errors.ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})
    exhausted_client.models.generate_content.side_effect = rate_limit

    with (
        patch(
            "lib.genai.fallback.create_genai_client",
            return_value=exhausted_client,
        ),
        pytest.raises(genai_errors.ClientError),
    ):
        generate_content_with_fallback(
            settings=settings,
            model="gemini-2.5-flash",
            contents="hello",
            config=config,
        )


def test_generate_content_with_fallback_non_429_does_not_cascade() -> None:
    settings = _vertex_settings()
    config = types.GenerateContentConfig(temperature=0)
    failing_client = MagicMock()
    failing_client.models.generate_content.side_effect = ValueError("bad request")

    with (
        patch(
            "lib.genai.fallback.create_genai_client",
            return_value=failing_client,
        ),
        pytest.raises(ValueError, match="bad request"),
    ):
        generate_content_with_fallback(
            settings=settings,
            model="gemini-2.5-flash",
            contents="hello",
            config=config,
        )

    assert failing_client.models.generate_content.call_count == 1
