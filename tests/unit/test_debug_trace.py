"""Unit tests for local debug trace helpers."""

from __future__ import annotations

import logging

import pytest

from lib.debug.trace import (
    configure_dev_logging,
    is_debug_trace_enabled,
    summarize_node_update,
    trace_node_update,
    trace_turn_start,
)


def test_is_debug_trace_enabled_defaults_on_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEBUG_TRACE", raising=False)
    monkeypatch.setenv("APP_ENV", "development")
    assert is_debug_trace_enabled() is True


def test_is_debug_trace_disabled_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEBUG_TRACE", raising=False)
    monkeypatch.setenv("APP_ENV", "production")
    assert is_debug_trace_enabled() is False


def test_debug_trace_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG_TRACE", "1")
    assert is_debug_trace_enabled() is True

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("DEBUG_TRACE", "0")
    assert is_debug_trace_enabled() is False


def test_summarize_node_update_shapes_mcp_results() -> None:
    summary = summarize_node_update(
        "call_mcp_tools",
        {
            "tool_call_count": 1,
            "tool_results": {
                "kapruka_search_products": {
                    "products": [{"name": "Rose Bouquet", "id": "p1"}],
                },
                "kapruka_track_order": {
                    "error": "order_not_found",
                    "message": "Missing order",
                },
            },
        },
    )
    assert summary["tool_call_count"] == 1
    assert summary["tool_results"]["kapruka_search_products"]["products"] == 1
    assert summary["tool_results"]["kapruka_track_order"]["error"] == "order_not_found"


def test_trace_turn_start_emits_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    configure_dev_logging()
    with caplog.at_level(logging.INFO, logger="agentic.trace"):
        trace_turn_start(
            thread_id="thread-1",
            message="birthday cakes",
            currency="LKR",
        )
    assert any("CHAT TURN ▶ START" in record.message for record in caplog.records)
    assert any("birthday cakes" in record.message for record in caplog.records)


def test_configure_dev_logging_silences_third_party_noise() -> None:
    configure_dev_logging()
    for name in ("httpcore", "httpx", "google_genai", "mcp"):
        assert logging.getLogger(name).level == logging.WARNING


def test_trace_node_update_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG_TRACE", "0")
    with caplog.at_level(logging.INFO, logger="agentic.trace"):
        trace_node_update("analyze_intent", {"intent": "discovery"})
    assert not caplog.records
