"""Unit tests for CrossEncoder reranker service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from lib.embeddings.reranker import (
    RERANKER_MODEL,
    CrossEncoderService,
    get_reranker,
)


def _mock_cross_encoder(*, scores: list[float] | None = None) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = np.array(scores or [0.82, 0.31])
    return model


def test_score_pairs_returns_float_scores() -> None:
    """Mocked cross-encoder returns one relevance score per text."""
    service = CrossEncoderService(model=_mock_cross_encoder(scores=[0.91, 0.44, 0.12]))

    scores = service.score_pairs("birthday cake", ["Cakes", "Flowers", "Electronics"])

    assert scores == [0.91, 0.44, 0.12]
    model = service._model
    assert model is not None
    model.predict.assert_called_once_with(
        [
            ("birthday cake", "Cakes"),
            ("birthday cake", "Flowers"),
            ("birthday cake", "Electronics"),
        ]
    )


def test_score_pairs_empty_texts_returns_empty_list() -> None:
    service = CrossEncoderService(model=_mock_cross_encoder())

    assert service.score_pairs("birthday cake", []) == []


def test_score_pairs_strips_query_whitespace() -> None:
    service = CrossEncoderService(model=_mock_cross_encoder(scores=[0.5]))

    service.score_pairs("  birthday cake  ", ["Cakes"])

    model = service._model
    assert model is not None
    model.predict.assert_called_once_with([("birthday cake", "Cakes")])


def test_lazy_model_load_uses_default_model_name() -> None:
    service = CrossEncoderService()

    with patch("lib.embeddings.reranker.CrossEncoder") as cross_encoder_cls:
        cross_encoder_cls.return_value = _mock_cross_encoder(scores=[0.7])
        scores = service.score_pairs("query", ["doc"])

    cross_encoder_cls.assert_called_once_with(RERANKER_MODEL)
    assert scores == [0.7]


def test_get_reranker_returns_singleton() -> None:
    import lib.embeddings.reranker as reranker_module

    reranker_module._reranker = None
    first = get_reranker()
    second = get_reranker()

    assert first is second
