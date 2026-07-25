"""NVIDIA NIM structured completion engine with rate limiting and retry."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel

from app.config import Settings, get_settings
from lib.genai.client import (
    NimKeyRole,
    create_nvidia_client,
    has_backup_nvidia_client,
)
from lib.genai.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Transient transport errors worth retrying (plus RateLimitError handled separately).
_TRANSIENT_ERRORS = (APITimeoutError, APIConnectionError)

# Absolute monotonic deadline for the current chat turn (set by streaming layer).
_turn_deadline_monotonic: ContextVar[float | None] = ContextVar(
    "nim_turn_deadline_monotonic",
    default=None,
)

# ── Markdown fence stripper ──────────────────────────────────────────────────
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)```",
    re.DOTALL,
)

_concurrency_semaphores: dict[int, asyncio.Semaphore] = {}


@contextmanager
def turn_deadline(seconds: float) -> Iterator[None]:
    """Bound NIM retries to a wall-clock deadline relative to now."""
    token = _turn_deadline_monotonic.set(time.monotonic() + max(0.0, seconds))
    try:
        yield
    finally:
        _turn_deadline_monotonic.reset(token)


def seconds_until_deadline() -> float | None:
    """Seconds remaining before the chat-turn wall-clock deadline, if set."""
    deadline = _turn_deadline_monotonic.get()
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _seconds_until_deadline() -> float | None:
    """Backward-compatible alias for :func:`seconds_until_deadline`."""
    return seconds_until_deadline()


def _should_abort_retry(*, reserve_seconds: float) -> bool:
    """True when fewer than reserve_seconds remain before the turn deadline."""
    remaining = seconds_until_deadline()
    if remaining is None:
        return False
    return remaining < reserve_seconds


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping a JSON payload.

    GLM-5.2 occasionally wraps structured responses in ```json ... ```.
    This utility extracts the inner content for safe Pydantic parsing.
    """
    text = text.strip()
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def _build_schema_instruction(schema_class: type[BaseModel]) -> str:
    """Generate a system-level instruction that embeds the JSON schema."""
    schema = schema_class.model_json_schema()
    return (
        "You MUST respond with valid JSON only — no markdown, no explanation, "
        "no code fences. Your response must conform exactly to this JSON schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


_override_generate_content = None


def set_override_generate_content(func: Any) -> None:
    """Set a global override for generate_content (used exclusively for testing)."""
    global _override_generate_content
    _override_generate_content = func


def reset_concurrency_limiter() -> None:
    """Drop cached concurrency semaphores (for tests)."""
    _concurrency_semaphores.clear()


def _get_concurrency_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    """Return a process-wide semaphore capped at ``max_concurrent`` NIM calls."""
    if max_concurrent not in _concurrency_semaphores:
        _concurrency_semaphores[max_concurrent] = asyncio.Semaphore(max_concurrent)
    return _concurrency_semaphores[max_concurrent]


def _retry_after_delay(exc: RateLimitError, max_delay: float) -> float | None:
    """Parse the Retry-After header (seconds) from a NIM 429 response."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if isinstance(raw, str) and raw.strip().isdigit():
        return min(max_delay, max(1.0, float(raw.strip())))
    return None


def _backoff_delay(attempt: int, *, base_delay: float, max_delay: float) -> float:
    return min(max_delay, base_delay * (2**attempt))


async def _complete_with_client(
    *,
    client: OpenAI,
    role: NimKeyRole,
    resolved_model: str,
    request_messages: list[dict[str, Any]],
    response_schema: type[T] | None,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    deadline_reserve_seconds: float = 15.0,
) -> dict[str, Any] | T:
    """Run chat completion against one NIM client with rate limit + retry."""
    limiter = get_rate_limiter(role=role)
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        if attempt > 0 and _should_abort_retry(reserve_seconds=deadline_reserve_seconds):
            logger.warning(
                "generate_content: aborting %s retries — turn deadline reserve exhausted",
                role,
            )
            break
        await limiter.acquire()
        try:
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=resolved_model,
                messages=request_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                seed=seed,
            )

            raw_content = completion.choices[0].message.content or ""

            if response_schema is not None:
                cleaned = strip_markdown_fences(raw_content)
                try:
                    return response_schema.model_validate_json(cleaned)
                except Exception:
                    logger.warning(
                        "generate_content: JSON parse failed on %s attempt %d, raw=%r",
                        role,
                        attempt + 1,
                        raw_content[:500],
                    )
                    if attempt < max_retries - 1:
                        if _should_abort_retry(reserve_seconds=deadline_reserve_seconds):
                            last_exc = ValueError(f"JSON parse failed: {raw_content[:200]}")
                            break
                        last_exc = ValueError(f"JSON parse failed: {raw_content[:200]}")
                        delay = _backoff_delay(
                            attempt, base_delay=base_delay, max_delay=max_delay
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise

            return {
                "content": raw_content,
                "role": completion.choices[0].message.role,
            }

        except RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                if _should_abort_retry(reserve_seconds=deadline_reserve_seconds):
                    logger.warning(
                        "generate_content: aborting %s 429 retries — turn deadline reserve exhausted",
                        role,
                    )
                    break
                backoff = _backoff_delay(
                    attempt, base_delay=base_delay, max_delay=max_delay
                )
                retry_after = _retry_after_delay(exc, max_delay)
                delay = max(backoff, retry_after) if retry_after is not None else backoff
                logger.warning(
                    "generate_content: NVIDIA NIM 429 on %s attempt %d; retrying in %.1fs",
                    role,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "generate_content: NVIDIA NIM 429 exhausted %d retries on %s",
                    max_retries,
                    role,
                )

        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                if _should_abort_retry(reserve_seconds=deadline_reserve_seconds):
                    logger.warning(
                        "generate_content: aborting %s %s retries — turn deadline reserve exhausted",
                        role,
                        type(exc).__name__,
                    )
                    break
                delay = _backoff_delay(
                    attempt, base_delay=base_delay, max_delay=max_delay
                )
                logger.warning(
                    "generate_content: NVIDIA NIM %s on %s attempt %d; retrying in %.1fs",
                    type(exc).__name__,
                    role,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "generate_content: NVIDIA NIM %s exhausted %d retries on %s",
                    type(exc).__name__,
                    max_retries,
                    role,
                )

    assert last_exc is not None
    raise last_exc


async def generate_content(
    *,
    model: str | None = None,
    messages: list[dict[str, Any]],
    response_schema: type[T] | None = None,
    temperature: float = 1.0,
    max_tokens: int = 16384,
    settings: Settings | None = None,
    seed: int | None = 42,
) -> dict[str, Any] | T:
    """Call NVIDIA NIM chat completion with rate limiting and retry.

    Parameters
    ----------
    model:
        NVIDIA NIM model name (defaults to ``Settings.nvidia_llm_model``).
    messages:
        OpenAI-format message list ``[{"role": ..., "content": ...}]``.
    response_schema:
        Optional Pydantic model class. When provided, the JSON schema is
        injected into the system prompt and the raw text response is parsed
        and validated via ``model_validate_json()``.
    temperature:
        Sampling temperature (0.0–2.0).
    max_tokens:
        Maximum tokens in the response.
    settings:
        Explicit settings override (for DI / tests).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Parsed Pydantic model instance when ``response_schema`` is provided,
    otherwise a dict with ``{"content": str, "role": str}``.
    """
    if _override_generate_content is not None:
        # Test hook: bypass the live NIM client entirely.
        return await _override_generate_content(
            model=model,
            messages=messages,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            settings=settings,
            seed=seed,
        )

    cfg = settings or get_settings()
    resolved_model = model or cfg.nvidia_llm_model
    base_delay = float(cfg.nvidia_retry_base_delay)
    max_delay = float(cfg.nvidia_retry_max_delay)
    primary_retries = int(cfg.nvidia_max_retries)
    backup_retries = int(cfg.nvidia_backup_max_retries)
    max_concurrent = int(cfg.nvidia_max_concurrent)
    deadline_reserve = float(getattr(cfg, "nvidia_deadline_reserve_seconds", 15.0))

    # Inject schema instruction into system prompt if needed
    request_messages = list(messages)
    if response_schema is not None:
        schema_instruction = _build_schema_instruction(response_schema)
        # Prepend or merge with existing system message
        if request_messages and request_messages[0].get("role") == "system":
            request_messages[0] = {
                "role": "system",
                "content": request_messages[0]["content"] + "\n\n" + schema_instruction,
            }
        else:
            request_messages.insert(0, {"role": "system", "content": schema_instruction})

    semaphore = _get_concurrency_semaphore(max_concurrent)
    async with semaphore:
        primary_client = create_nvidia_client(settings=cfg, role="primary")
        try:
            return await _complete_with_client(
                client=primary_client,
                role="primary",
                resolved_model=resolved_model,
                request_messages=request_messages,
                response_schema=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                max_retries=primary_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                deadline_reserve_seconds=deadline_reserve,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError):
            if not has_backup_nvidia_client(settings=cfg):
                raise
            if _should_abort_retry(reserve_seconds=deadline_reserve):
                raise
            logger.warning(
                "NIM primary exhausted; failing over to backup key",
            )
            backup_client = create_nvidia_client(settings=cfg, role="backup")
            return await _complete_with_client(
                client=backup_client,
                role="backup",
                resolved_model=resolved_model,
                request_messages=request_messages,
                response_schema=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                max_retries=backup_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                deadline_reserve_seconds=deadline_reserve,
            )
