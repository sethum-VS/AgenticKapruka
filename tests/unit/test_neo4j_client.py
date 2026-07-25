"""Integration tests for async Neo4j client wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from lib.neo4j.client import Neo4jClient
from neo4j import AsyncGraphDatabase


class _MockAsyncResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def data(self) -> list[dict[str, Any]]:
        return self._rows


class _MockAsyncSession:
    async def run(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _MockAsyncResult:
        del kwargs
        if cypher == "RETURN 1 AS ok":
            return _MockAsyncResult([{"ok": 1}])
        if cypher == "RETURN $value AS value":
            return _MockAsyncResult([{"value": (parameters or {}).get("value")}])
        return _MockAsyncResult([])

    async def __aenter__(self) -> _MockAsyncSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _MockAsyncDriver:
    def __init__(self) -> None:
        self.verify_called = False
        self.closed = False

    async def verify_connectivity(self) -> None:
        self.verify_called = True

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncIterator[_MockAsyncSession]:
        del kwargs
        yield _MockAsyncSession()

    async def close(self) -> None:
        self.closed = True


async def test_neo4j_client_health_check_with_mock_driver() -> None:
    """health_check returns True when RETURN 1 AS ok succeeds."""
    mock_driver = _MockAsyncDriver()
    client = Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=mock_driver,  # type: ignore[arg-type]
    )

    assert await client.health_check() is True

    await client.close()
    assert mock_driver.closed is True


async def test_neo4j_client_execute_returns_record_dicts() -> None:
    """execute() maps Cypher records to plain dicts."""
    mock_driver = _MockAsyncDriver()
    client = Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=mock_driver,  # type: ignore[arg-type]
    )

    rows = await client.execute("RETURN $value AS value", {"value": "kapruka"})

    assert rows == [{"value": "kapruka"}]

    await client.close()


async def test_neo4j_client_execute_retries_connection_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ConnectionResetError triggers reconnect and a successful retry."""
    attempts = {"count": 0}
    drivers_created: list[_MockAsyncDriver] = []

    class _FlakySession(_MockAsyncSession):
        async def run(
            self,
            cypher: str,
            parameters: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> _MockAsyncResult:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ConnectionResetError("connection reset by peer")
            return await super().run(cypher, parameters, **kwargs)

    class _FlakyDriver(_MockAsyncDriver):
        @asynccontextmanager
        async def session(self, **kwargs: Any) -> AsyncIterator[_FlakySession]:
            del kwargs
            yield _FlakySession()

    def mock_driver_factory(
        uri: str,
        *,
        auth: tuple[str, str] | None = None,
        **config: Any,
    ) -> _MockAsyncDriver:
        del uri, auth, config
        driver = _FlakyDriver()
        drivers_created.append(driver)
        return driver

    monkeypatch.setattr(AsyncGraphDatabase, "driver", mock_driver_factory)
    monkeypatch.setattr("lib.neo4j.client.asyncio.sleep", _async_noop)

    first_driver = _FlakyDriver()
    client = Neo4jClient(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        driver=first_driver,  # type: ignore[arg-type]
    )

    rows = await client.execute("RETURN $value AS value", {"value": "kapruka"})

    assert rows == [{"value": "kapruka"}]
    assert attempts["count"] == 2
    assert first_driver.closed is True
    assert len(drivers_created) == 1
    assert drivers_created[0].verify_called is True


async def _async_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


async def test_neo4j_client_connect_verifies_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect() builds driver via AsyncGraphDatabase.driver and verifies connectivity."""
    mock_driver = _MockAsyncDriver()
    captured: dict[str, Any] = {}

    def mock_driver_factory(
        uri: str,
        *,
        auth: tuple[str, str] | None = None,
        **config: Any,
    ) -> _MockAsyncDriver:
        captured["uri"] = uri
        captured["auth"] = auth
        captured["config"] = config
        return mock_driver

    monkeypatch.setattr(AsyncGraphDatabase, "driver", mock_driver_factory)

    client = await Neo4jClient.connect(
        "bolt+s://aura.databases.neo4j.io",
        "neo4j",
        "secret",
    )

    assert captured["uri"] == "bolt+s://aura.databases.neo4j.io"
    assert captured["auth"] == ("neo4j", "secret")
    assert captured["config"]["connection_timeout"] == 30.0
    assert mock_driver.verify_called is True
    assert await client.health_check() is True

    await client.close()
    assert mock_driver.closed is True
    await client.close()  # idempotent
