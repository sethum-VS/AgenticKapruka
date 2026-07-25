"""Thread-safe token-bucket rate limiter for NVIDIA NIM free-tier safety."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Literal

from app.config import get_settings

NimLimiterRole = Literal["primary", "backup"]


class TokenBucketRateLimiter:
    """Token-bucket that refills at ``rpm`` tokens per minute.

    Call ``await acquire()`` before every NVIDIA NIM API request to
    proactively stay under the free-tier ~40 RPM soft limit.
    """

    def __init__(self, rpm: int = 30) -> None:
        self._rpm = max(1, rpm)
        self._interval = 60.0 / self._rpm  # seconds between tokens
        self._tokens = float(self._rpm)  # start full
        self._max_tokens = float(self._rpm)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed / self._interval
        self._tokens = min(self._max_tokens, self._tokens + new_tokens)
        self._last_refill = now

    async def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            wait_time: float | None = None
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time until next token
                wait_time = self._interval * (1.0 - self._tokens)
            await asyncio.sleep(wait_time)


_limiters: dict[NimLimiterRole, TokenBucketRateLimiter] = {}
_limiter_lock = threading.Lock()


def get_rate_limiter(*, role: NimLimiterRole = "primary") -> TokenBucketRateLimiter:
    """Return the singleton rate limiter for the given NIM key role."""
    existing = _limiters.get(role)
    if existing is not None:
        return existing
    with _limiter_lock:
        existing = _limiters.get(role)
        if existing is not None:
            return existing
        cfg = get_settings()
        limiter = TokenBucketRateLimiter(rpm=cfg.nvidia_rate_limit_rpm)
        _limiters[role] = limiter
        return limiter


def reset_rate_limiter() -> None:
    """Drop cached limiters (for tests)."""
    _limiters.clear()
