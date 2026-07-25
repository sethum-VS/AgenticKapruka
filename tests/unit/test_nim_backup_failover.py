"""Unit tests for NVIDIA NIM primary → backup API key failover."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APITimeoutError, RateLimitError

from lib.genai.client import reset_client
from lib.genai.completions import (
    generate_content,
    reset_concurrency_limiter,
    set_override_generate_content,
)
from lib.genai.rate_limiter import reset_rate_limiter


def _rate_limit_error() -> RateLimitError:
    response = MagicMock()
    response.headers = {}
    response.status_code = 429
    return RateLimitError("Rate limit exceeded", response=response, body=None)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=MagicMock())


def _fake_settings(*, backup: str | None = "nvapi-backup-key") -> SimpleNamespace:
    return SimpleNamespace(
        nvidia_api_key="nvapi-primary-key",
        nvidia_api_key_backup=backup,
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_llm_model="z-ai/glm-5.2",
        nvidia_rate_limit_rpm=30,
        nvidia_max_concurrent=2,
        nvidia_retry_base_delay=3.0,
        nvidia_retry_max_delay=60.0,
        nvidia_max_retries=4,
        nvidia_backup_max_retries=2,
        nvidia_http_timeout=30.0,
    )


@pytest.fixture(autouse=True)
def _reset_nim_singletons() -> None:
    set_override_generate_content(None)
    reset_client()
    reset_rate_limiter()
    reset_concurrency_limiter()
    yield
    set_override_generate_content(None)
    reset_client()
    reset_rate_limiter()
    reset_concurrency_limiter()


@pytest.mark.asyncio
async def test_generate_content_fails_over_to_backup_on_primary_429() -> None:
    primary = MagicMock()
    backup = MagicMock()
    primary.chat.completions.create.side_effect = _rate_limit_error()
    backup_completion = MagicMock()
    backup_completion.choices = [
        MagicMock(message=MagicMock(content="from backup", role="assistant")),
    ]
    backup.chat.completions.create.return_value = backup_completion

    def _client_for_role(*, settings=None, role: str = "primary"):
        return primary if role == "primary" else backup

    settings = _fake_settings()
    with (
        patch("lib.genai.completions.get_settings", return_value=settings),
        patch("lib.genai.completions.create_nvidia_client", side_effect=_client_for_role),
        patch("lib.genai.completions.has_backup_nvidia_client", return_value=True),
        patch("lib.genai.completions.get_rate_limiter") as mock_limiter,
        patch("lib.genai.completions.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_limiter.return_value.acquire = AsyncMock()
        result = await generate_content(
            messages=[{"role": "user", "content": "hi"}],
            settings=settings,  # type: ignore[arg-type]
        )

    assert result["content"] == "from backup"
    assert primary.chat.completions.create.call_count == 4
    assert backup.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_generate_content_raises_when_primary_and_backup_429() -> None:
    primary = MagicMock()
    backup = MagicMock()
    primary.chat.completions.create.side_effect = _rate_limit_error()
    backup.chat.completions.create.side_effect = _rate_limit_error()

    def _client_for_role(*, settings=None, role: str = "primary"):
        return primary if role == "primary" else backup

    settings = _fake_settings()
    with (
        patch("lib.genai.completions.get_settings", return_value=settings),
        patch("lib.genai.completions.create_nvidia_client", side_effect=_client_for_role),
        patch("lib.genai.completions.has_backup_nvidia_client", return_value=True),
        patch("lib.genai.completions.get_rate_limiter") as mock_limiter,
        patch("lib.genai.completions.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RateLimitError),
    ):
        mock_limiter.return_value.acquire = AsyncMock()
        await generate_content(
            messages=[{"role": "user", "content": "hi"}],
            settings=settings,  # type: ignore[arg-type]
        )

    assert primary.chat.completions.create.call_count == 4
    assert backup.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_generate_content_no_failover_without_backup_key() -> None:
    primary = MagicMock()
    primary.chat.completions.create.side_effect = _rate_limit_error()

    settings = _fake_settings(backup=None)
    with (
        patch("lib.genai.completions.get_settings", return_value=settings),
        patch(
            "lib.genai.completions.create_nvidia_client",
            return_value=primary,
        ),
        patch("lib.genai.completions.has_backup_nvidia_client", return_value=False),
        patch("lib.genai.completions.get_rate_limiter") as mock_limiter,
        patch("lib.genai.completions.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RateLimitError),
    ):
        mock_limiter.return_value.acquire = AsyncMock()
        await generate_content(
            messages=[{"role": "user", "content": "hi"}],
            settings=settings,  # type: ignore[arg-type]
        )

    assert primary.chat.completions.create.call_count == 4


@pytest.mark.asyncio
async def test_generate_content_retries_timeout_then_fails_over() -> None:
    primary = MagicMock()
    backup = MagicMock()
    primary.chat.completions.create.side_effect = _timeout_error()
    backup_completion = MagicMock()
    backup_completion.choices = [
        MagicMock(message=MagicMock(content="from backup after timeout", role="assistant")),
    ]
    backup.chat.completions.create.return_value = backup_completion

    def _client_for_role(*, settings=None, role: str = "primary"):
        return primary if role == "primary" else backup

    settings = _fake_settings()
    with (
        patch("lib.genai.completions.get_settings", return_value=settings),
        patch("lib.genai.completions.create_nvidia_client", side_effect=_client_for_role),
        patch("lib.genai.completions.has_backup_nvidia_client", return_value=True),
        patch("lib.genai.completions.get_rate_limiter") as mock_limiter,
        patch("lib.genai.completions.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_limiter.return_value.acquire = AsyncMock()
        result = await generate_content(
            messages=[{"role": "user", "content": "hi"}],
            settings=settings,  # type: ignore[arg-type]
        )

    assert result["content"] == "from backup after timeout"
    assert primary.chat.completions.create.call_count == 4
    assert backup.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_generate_content_respects_concurrency_semaphore() -> None:
    """Only nvidia_max_concurrent calls run the completion path at once."""
    settings = _fake_settings()
    settings.nvidia_max_concurrent = 1
    release_first = asyncio.Event()
    entered = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def _slow_complete(**kwargs):
        nonlocal entered, max_seen
        async with lock:
            entered += 1
            max_seen = max(max_seen, entered)
        await release_first.wait()
        async with lock:
            entered -= 1
        return {"content": "ok", "role": "assistant"}

    with (
        patch("lib.genai.completions.get_settings", return_value=settings),
        patch(
            "lib.genai.completions._complete_with_client",
            new_callable=AsyncMock,
            side_effect=_slow_complete,
        ),
        patch("lib.genai.completions.create_nvidia_client", return_value=MagicMock()),
        patch("lib.genai.completions.has_backup_nvidia_client", return_value=False),
    ):
        task1 = asyncio.create_task(
            generate_content(
                messages=[{"role": "user", "content": "a"}],
                settings=settings,  # type: ignore[arg-type]
            )
        )
        await asyncio.sleep(0)
        task2 = asyncio.create_task(
            generate_content(
                messages=[{"role": "user", "content": "b"}],
                settings=settings,  # type: ignore[arg-type]
            )
        )
        await asyncio.sleep(0.05)
        assert max_seen == 1
        release_first.set()
        results = await asyncio.gather(task1, task2)

    assert [r["content"] for r in results] == ["ok", "ok"]
    assert max_seen == 1
