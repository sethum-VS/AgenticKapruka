"""Tests for scripts/bootstrap_neo4j.py."""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.config import Settings
from scripts import bootstrap_neo4j


def _bootstrap_settings() -> Settings:
    return Settings(
        redis_url="redis://localhost:6379/0",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test-password",
        zep_api_key="zep-test-key",
        gcp_project_id="test-project",
        gcp_location="us-central1",
        session_secret="x" * 32,
        _env_file=None,
    )


@pytest.mark.asyncio
async def test_bootstrap_runs_all_steps_in_order() -> None:
    """Full bootstrap runs migrate → ingest → embed → index → verify."""
    client = MagicMock()
    call_order: list[str] = []

    async def track_migrate(c: object) -> bool:
        call_order.append("migrate")
        return True

    async def track_ingest(c: object, depth: int) -> bool:
        call_order.append(f"ingest:{depth}")
        return True

    async def track_embed(c: object) -> bool:
        call_order.append("embed")
        return True

    async def track_index(c: object) -> bool:
        call_order.append("index")
        return True

    async def track_verify(c: object) -> bool:
        call_order.append("verify")
        return True

    with (
        patch.object(bootstrap_neo4j, "get_settings", return_value=_bootstrap_settings()),
        patch.object(bootstrap_neo4j, "Neo4jClient") as mock_client_cls,
        patch.object(bootstrap_neo4j, "_run_migrate", side_effect=track_migrate),
        patch.object(bootstrap_neo4j, "_run_ingest", side_effect=track_ingest),
        patch.object(bootstrap_neo4j, "_run_embed", side_effect=track_embed),
        patch.object(bootstrap_neo4j, "_run_index", side_effect=track_index),
        patch.object(bootstrap_neo4j, "_verify", side_effect=track_verify),
    ):
        mock_client_cls.connect = AsyncMock(return_value=client)
        client.close = AsyncMock()
        args = argparse.Namespace(
            skip_migrate=False,
            skip_ingest=False,
            skip_embed=False,
            skip_index=False,
            depth=2,
        )
        code = await bootstrap_neo4j._run(args)

    assert code == 0
    assert call_order == ["migrate", "ingest:2", "embed", "index", "verify"]
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_bootstrap_honors_skip_flags() -> None:
    """Skip flags omit the corresponding bootstrap steps."""
    client = MagicMock()
    call_order: list[str] = []

    with (
        patch.object(bootstrap_neo4j, "get_settings", return_value=_bootstrap_settings()),
        patch.object(bootstrap_neo4j, "Neo4jClient") as mock_client_cls,
        patch.object(
            bootstrap_neo4j,
            "_run_migrate",
            side_effect=lambda c: call_order.append("migrate") or True,
        ),
        patch.object(
            bootstrap_neo4j,
            "_run_ingest",
            side_effect=lambda c, d: call_order.append("ingest") or True,
        ),
        patch.object(
            bootstrap_neo4j,
            "_run_embed",
            side_effect=lambda c: call_order.append("embed") or True,
        ),
        patch.object(
            bootstrap_neo4j,
            "_run_index",
            side_effect=lambda c: call_order.append("index") or True,
        ),
        patch.object(
            bootstrap_neo4j,
            "_verify",
            side_effect=lambda c: call_order.append("verify") or True,
        ),
    ):
        mock_client_cls.connect = AsyncMock(return_value=client)
        client.close = AsyncMock()
        args = argparse.Namespace(
            skip_migrate=True,
            skip_ingest=True,
            skip_embed=True,
            skip_index=True,
            depth=2,
        )
        code = await bootstrap_neo4j._run(args)

    assert code == 0
    assert call_order == ["verify"]


@pytest.mark.asyncio
async def test_bootstrap_exits_nonzero_when_verify_fails() -> None:
    """Verification failure returns exit code 1."""
    client = MagicMock()

    with (
        patch.object(bootstrap_neo4j, "get_settings", return_value=_bootstrap_settings()),
        patch.object(bootstrap_neo4j, "Neo4jClient") as mock_client_cls,
        patch.object(bootstrap_neo4j, "_run_migrate", AsyncMock(return_value=True)),
        patch.object(bootstrap_neo4j, "_run_ingest", AsyncMock(return_value=True)),
        patch.object(bootstrap_neo4j, "_run_embed", AsyncMock(return_value=True)),
        patch.object(bootstrap_neo4j, "_run_index", AsyncMock(return_value=True)),
        patch.object(bootstrap_neo4j, "_verify", AsyncMock(return_value=False)),
    ):
        mock_client_cls.connect = AsyncMock(return_value=client)
        client.close = AsyncMock()
        args = argparse.Namespace(
            skip_migrate=False,
            skip_ingest=False,
            skip_embed=False,
            skip_index=False,
            depth=2,
        )
        code = await bootstrap_neo4j._run(args)

    assert code == 1


@pytest.mark.asyncio
async def test_bootstrap_exits_nonzero_when_index_step_fails() -> None:
    """Missing vector index after create returns exit code 1."""
    client = MagicMock()

    with (
        patch.object(bootstrap_neo4j, "get_settings", return_value=_bootstrap_settings()),
        patch.object(bootstrap_neo4j, "Neo4jClient") as mock_client_cls,
        patch.object(bootstrap_neo4j, "_run_migrate", AsyncMock(return_value=True)),
        patch.object(bootstrap_neo4j, "_run_ingest", AsyncMock(return_value=True)),
        patch.object(bootstrap_neo4j, "_run_embed", AsyncMock(return_value=True)),
        patch.object(bootstrap_neo4j, "_run_index", AsyncMock(return_value=False)),
    ):
        mock_client_cls.connect = AsyncMock(return_value=client)
        client.close = AsyncMock()
        args = argparse.Namespace(
            skip_migrate=False,
            skip_ingest=False,
            skip_embed=False,
            skip_index=False,
            depth=2,
        )
        code = await bootstrap_neo4j._run(args)

    assert code == 1
