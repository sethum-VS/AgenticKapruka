"""Cross-encoder reranker for post-retrieval query–node relevance scoring."""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderService:
    """Lazy-loaded cross-encoder for scoring query–text pairs."""

    def __init__(
        self,
        *,
        model_name: str = RERANKER_MODEL,
        model: CrossEncoder | None = None,
    ) -> None:
        self._model_name = model_name
        self._model = model

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.debug("Loading cross-encoder model %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def preload(self) -> None:
        """Eagerly load the cross-encoder model (call from app startup)."""
        self._get_model()

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Score each text against the query; higher values mean stronger relevance."""
        if not texts:
            return []

        stripped_query = query.strip()
        pairs = [(stripped_query, text) for text in texts]
        raw_scores = self._get_model().predict(pairs)
        return [float(score) for score in raw_scores]


_reranker: CrossEncoderService | None = None


def get_reranker() -> CrossEncoderService:
    """Return the process-wide CrossEncoderService singleton."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderService()
    return _reranker


def preload_reranker() -> None:
    """Load the cross-encoder model during app startup to avoid first-request latency."""
    get_reranker().preload()
