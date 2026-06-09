"""Tests for scripts/embed_ontology.py CLI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from scripts import embed_ontology as embed_script


@pytest.mark.asyncio
async def test_embed_script_creates_vector_index_after_embed() -> None:
    """embed_ontology.py creates and verifies the Category vector index."""
    client = MagicMock()
    client.close = AsyncMock()

    with (
        patch.object(embed_script, "get_settings"),
        patch.object(embed_script, "Neo4jClient") as mock_client_cls,
        patch.object(embed_script, "embed_ontology_nodes", AsyncMock()) as mock_embed,
        patch.object(embed_script, "has_category_embeddings", AsyncMock(return_value=True)),
        patch.object(
            embed_script,
            "create_category_vector_index",
            AsyncMock(),
        ) as mock_create_index,
        patch.object(embed_script, "has_category_vector_index", AsyncMock(return_value=True)),
        patch.object(embed_script, "count_nodes_with_embedding", AsyncMock(return_value=3)),
    ):
        mock_embed.return_value = MagicMock(nodes_embedded=3, batches_written=1)
        mock_client_cls.connect = AsyncMock(return_value=client)
        code = await embed_script._run(force_reembed=False)

    assert code == 0
    mock_create_index.assert_awaited_once_with(client)


@pytest.mark.asyncio
async def test_embed_script_force_reembed_clears_existing_vectors() -> None:
    client = MagicMock()
    client.close = AsyncMock()

    with (
        patch.object(embed_script, "get_settings"),
        patch.object(embed_script, "Neo4jClient") as mock_client_cls,
        patch.object(
            embed_script,
            "clear_ontology_embeddings",
            AsyncMock(return_value=12),
        ) as mock_clear,
        patch.object(embed_script, "embed_ontology_nodes", AsyncMock()) as mock_embed,
        patch.object(embed_script, "has_category_embeddings", AsyncMock(return_value=True)),
        patch.object(embed_script, "create_category_vector_index", AsyncMock()),
        patch.object(embed_script, "has_category_vector_index", AsyncMock(return_value=True)),
        patch.object(embed_script, "count_nodes_with_embedding", AsyncMock(return_value=3)),
    ):
        mock_embed.return_value = MagicMock(nodes_embedded=3, batches_written=1)
        mock_client_cls.connect = AsyncMock(return_value=client)
        code = await embed_script._run(force_reembed=True)

    assert code == 0
    mock_clear.assert_awaited_once_with(client)


@pytest.mark.asyncio
async def test_embed_script_fails_when_vector_index_missing() -> None:
    """embed_ontology.py exits non-zero when vector index verification fails."""
    client = MagicMock()
    client.close = AsyncMock()

    with (
        patch.object(embed_script, "get_settings"),
        patch.object(embed_script, "Neo4jClient") as mock_client_cls,
        patch.object(embed_script, "embed_ontology_nodes", AsyncMock()),
        patch.object(embed_script, "has_category_embeddings", AsyncMock(return_value=True)),
        patch.object(embed_script, "create_category_vector_index", AsyncMock()),
        patch.object(embed_script, "has_category_vector_index", AsyncMock(return_value=False)),
    ):
        mock_client_cls.connect = AsyncMock(return_value=client)
        code = await embed_script._run()

    assert code == 1
