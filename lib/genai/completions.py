"""NVIDIA NIM structured completion engine with rate limiting and retry."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, TypeVar

from openai import RateLimitError
from pydantic import BaseModel

from app.config import Settings, get_settings
from lib.genai.client import create_nvidia_client
from lib.genai.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ── Retry constants ──────────────────────────────────────────────────────────
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0  # seconds (2s, 4s, 8s, 16s exponential)
_RETRY_MAX_DELAY = 30.0  # cap any single backoff (incl. Retry-After) at 30s

# ── Markdown fence stripper ──────────────────────────────────────────────────
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)```",
    re.DOTALL,
)


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


def _retry_after_delay(exc: RateLimitError) -> float | None:
    """Parse the Retry-After header (seconds) from a NIM 429 response."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if isinstance(raw, str) and raw.strip().isdigit():
        return min(_RETRY_MAX_DELAY, max(1.0, float(raw.strip())))
    return None


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
    client = create_nvidia_client(settings=cfg)
    limiter = get_rate_limiter()

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

    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
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
                        "generate_content: JSON parse failed on attempt %d, raw=%r",
                        attempt + 1,
                        raw_content[:500],
                    )
                    if attempt < _MAX_RETRIES - 1:
                        last_exc = ValueError(f"JSON parse failed: {raw_content[:200]}")
                        delay = _RETRY_BASE_DELAY * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
                    raise

            return {
                "content": raw_content,
                "role": completion.choices[0].message.role,
            }

        except RateLimitError as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                backoff = min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * (2 ** attempt))
                # Respect a server-provided Retry-After when it is longer than our backoff.
                retry_after = _retry_after_delay(exc)
                delay = max(backoff, retry_after) if retry_after is not None else backoff
                logger.warning(
                    "generate_content: NVIDIA NIM 429 on attempt %d; retrying in %.1fs",
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "generate_content: NVIDIA NIM 429 exhausted %d retries",
                    _MAX_RETRIES,
                )

    assert last_exc is not None
    raise last_exc
