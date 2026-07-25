"""Async Neo4j driver wrapper with session management and health checks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from neo4j import AsyncDriver, AsyncGraphDatabase

logger = logging.getLogger(__name__)

_DEFAULT_CONNECTION_TIMEOUT = 30.0
_DEFAULT_MAX_CONNECTION_LIFETIME = 3600
_DEFAULT_QUERY_TIMEOUT = 8.0
_HEALTH_CHECK_CYPHER = "RETURN 1 AS ok"
_EXECUTE_MAX_ATTEMPTS = 3
_EXECUTE_RETRY_BASE_DELAY = 0.25

_TRANSIENT_NEO4J_ERRORS = (
    ConnectionResetError,
    ConnectionError,
    OSError,
    ServiceUnavailable,
    SessionExpired,
)


class Neo4jClient:
    """Wrap neo4j async driver with session helpers, verify, and health checks."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        driver: AsyncDriver | None = None,
        connection_timeout: float = _DEFAULT_CONNECTION_TIMEOUT,
        max_connection_lifetime: int = _DEFAULT_MAX_CONNECTION_LIFETIME,
        query_timeout: float = _DEFAULT_QUERY_TIMEOUT,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver_config: dict[str, Any] = {
            "connection_timeout": connection_timeout,
            "max_connection_lifetime": max_connection_lifetime,
        }
        self._query_timeout = query_timeout
        self._driver = driver

    @classmethod
    async def connect(
        cls,
        uri: str,
        user: str,
        password: str,
        *,
        connection_timeout: float = _DEFAULT_CONNECTION_TIMEOUT,
        max_connection_lifetime: int = _DEFAULT_MAX_CONNECTION_LIFETIME,
    ) -> Neo4jClient:
        """Create a connected client and verify AuraDB / local Neo4j connectivity."""
        instance = cls(
            uri,
            user,
            password,
            connection_timeout=connection_timeout,
            max_connection_lifetime=max_connection_lifetime,
        )
        instance._driver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            **instance._driver_config,
        )
        await instance._driver.verify_connectivity()
        return instance

    @property
    def driver(self) -> AsyncDriver:
        """Underlying neo4j async driver for graph modules."""
        if self._driver is None:
            msg = "Neo4jClient is not connected; call connect() first"
            raise RuntimeError(msg)
        return self._driver

    def _is_transient_error(self, exc: BaseException) -> bool:
        if isinstance(exc, _TRANSIENT_NEO4J_ERRORS):
            return True
        if isinstance(exc, Neo4jError):
            code = getattr(exc, "code", "") or ""
            return "TransientError" in str(code) or "ServiceUnavailable" in str(exc)
        return False

    async def _reconnect(self) -> None:
        """Close the current driver and open a fresh connection."""
        old_driver = self._driver
        self._driver = None
        if old_driver is not None:
            try:
                await old_driver.close()
            except Exception as close_exc:
                logger.debug("Neo4j driver close during reconnect failed: %s", close_exc)
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
            **self._driver_config,
        )
        await self._driver.verify_connectivity()
        logger.info("Neo4j driver reconnected after transient failure")

    async def execute(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> list[dict[str, Any]]:
        """Run Cypher and return each record as a plain dict.

        Retries transient connection failures (e.g. ConnectionResetError) with
        a short backoff and driver reconnect. Each attempt is capped by
        ``timeout`` (default ``query_timeout``) so Aura stalls cannot consume
        the full chat turn budget.
        """
        import time

        query_timeout = self._query_timeout if timeout is None else timeout
        last_exc: BaseException | None = None
        started = time.monotonic()
        for attempt in range(_EXECUTE_MAX_ATTEMPTS):
            try:
                async with asyncio.timeout(query_timeout):
                    async with self.driver.session() as session:
                        result = await session.run(cypher, params or {})
                        rows = await result.data()
                elapsed_ms = (time.monotonic() - started) * 1000
                logger.debug(
                    "Neo4j execute ok in %.0fms (attempt %d)",
                    elapsed_ms,
                    attempt + 1,
                )
                return rows
            except TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "Neo4j execute timed out after %.1fs on attempt %d/%d",
                    query_timeout,
                    attempt + 1,
                    _EXECUTE_MAX_ATTEMPTS,
                )
                if attempt >= _EXECUTE_MAX_ATTEMPTS - 1:
                    raise
                try:
                    await self._reconnect()
                except Exception as reconnect_exc:
                    logger.warning("Neo4j reconnect failed: %s", reconnect_exc)
                await asyncio.sleep(_EXECUTE_RETRY_BASE_DELAY * (2**attempt))
            except Exception as exc:
                last_exc = exc
                if not self._is_transient_error(exc) or attempt >= _EXECUTE_MAX_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "Neo4j execute transient failure on attempt %d/%d: %s",
                    attempt + 1,
                    _EXECUTE_MAX_ATTEMPTS,
                    exc,
                )
                try:
                    await self._reconnect()
                except Exception as reconnect_exc:
                    logger.warning("Neo4j reconnect failed: %s", reconnect_exc)
                await asyncio.sleep(_EXECUTE_RETRY_BASE_DELAY * (2**attempt))
        assert last_exc is not None
        raise last_exc

    async def health_check(self) -> bool:
        """Return True when Neo4j responds to RETURN 1 AS ok."""
        try:
            rows = await self.execute(_HEALTH_CHECK_CYPHER)
        except Exception as exc:
            logger.warning("Neo4j health check failed: %s", exc)
            return False
        return bool(rows) and rows[0].get("ok") == 1

    async def close(self) -> None:
        """Close the driver; safe to call multiple times."""
        if self._driver is None:
            return
        await self._driver.close()
        self._driver = None
        logger.debug("Neo4j driver closed")
