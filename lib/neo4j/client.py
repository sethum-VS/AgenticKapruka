"""Async Neo4j driver wrapper with session management and health checks."""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

logger = logging.getLogger(__name__)

_DEFAULT_CONNECTION_TIMEOUT = 30.0
_DEFAULT_MAX_CONNECTION_LIFETIME = 3600
_HEALTH_CHECK_CYPHER = "RETURN 1 AS ok"


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
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver_config: dict[str, Any] = {
            "connection_timeout": connection_timeout,
            "max_connection_lifetime": max_connection_lifetime,
        }
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

    async def execute(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run Cypher and return each record as a plain dict."""
        async with self.driver.session() as session:
            result = await session.run(cypher, params or {})
            return await result.data()

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
