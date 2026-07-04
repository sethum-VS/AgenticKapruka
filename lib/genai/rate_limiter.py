"""Thread-safe token-bucket rate limiter for NVIDIA NIM free-tier safety."""

from __future__ import annotations

import asyncio
import time
import threading

from app.config import get_settings


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


_limiter: TokenBucketRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> TokenBucketRateLimiter:
    """Return the singleton rate limiter (lazily created from settings)."""
    global _limiter
    if _limiter is not None:
        return _limiter
    with _limiter_lock:
        if _limiter is not None:
            return _limiter
        cfg = get_settings()
        _limiter = TokenBucketRateLimiter(rpm=cfg.nvidia_rate_limit_rpm)
        return _limiter


def reset_rate_limiter() -> None:
    """Drop cached limiter (for tests)."""
    global _limiter
    _limiter = None
