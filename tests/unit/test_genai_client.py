"""Tests for lib.genai.client factory."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings
from lib.genai.client import VERTEX_HTTP_OPTIONS, create_genai_client


def _vertex_settings() -> Settings:
    return Settings(
        redis_url="redis://localhost:6379/0",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test-password",
        zep_api_key="zep-test-key",
        gcp_project_id="test-project",
        gcp_location="us-central1",
        session_secret="x" * 32,
        gemini_backend="vertex",
        _env_file=None,
    )


def test_create_genai_client_uses_vertex_global_with_retry_options() -> None:
    settings = _vertex_settings()
    with patch("lib.genai.client.genai.Client") as mock_client:
        create_genai_client(settings=settings)
    mock_client.assert_called_once_with(
        vertexai=True,
        project="test-project",
        location="global",
        http_options=VERTEX_HTTP_OPTIONS,
    )
    retry = VERTEX_HTTP_OPTIONS.retry_options
    assert retry is not None
    assert retry.attempts == 8
    assert 429 in (retry.http_status_codes or [])


def test_create_genai_client_explicit_location_override() -> None:
    settings = _vertex_settings()
    with patch("lib.genai.client.genai.Client") as mock_client:
        create_genai_client(settings=settings, location="us-central1")
    mock_client.assert_called_once_with(
        vertexai=True,
        project="test-project",
        location="us-central1",
        http_options=VERTEX_HTTP_OPTIONS,
    )


def test_create_genai_client_api_key_backend() -> None:
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test-password",
        zep_api_key="zep-test-key",
        gcp_project_id="test-project",
        gcp_location="us-central1",
        session_secret="x" * 32,
        gemini_backend="api_key",
        google_api_key="test-api-key",
        _env_file=None,
    )
    with patch("lib.genai.client.genai.Client") as mock_client:
        create_genai_client(settings=settings)
    mock_client.assert_called_once_with(api_key="test-api-key")


def test_create_genai_client_explicit_api_key_override() -> None:
    with patch("lib.genai.client.genai.Client") as mock_client:
        create_genai_client(api_key="injected-key")
    mock_client.assert_called_once_with(api_key="injected-key")


def test_create_genai_client_api_key_backend_requires_key() -> None:
    settings = _vertex_settings().model_construct(
        gemini_backend="api_key",
        google_api_key=None,
    )
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        create_genai_client(settings=settings)


def test_settings_rejects_api_key_backend_without_key() -> None:
    with pytest.raises(ValidationError, match="GOOGLE_API_KEY"):
        Settings(
            redis_url="redis://localhost:6379/0",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test-password",
            zep_api_key="zep-test-key",
            gcp_project_id="test-project",
            gcp_location="us-central1",
            session_secret="x" * 32,
            gemini_backend="api_key",
            _env_file=None,
        )
