"""Unit tests for local debug trace helpers."""

from __future__ import annotations

import logging

import pytest

from lib.debug.trace import (
    configure_dev_logging,
    is_debug_trace_enabled,
    summarize_node_update,
    trace_agent_iteration,
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


def test_summarize_node_update_agent_loop_shape() -> None:
    """agent_loop branch reports iterations, tools, exit reason, and trace sizes only."""
    huge_product = {"id": "p1", "name": "Rose", "image_url": "https://example.com/x.jpg"}
    summary = summarize_node_update(
        "agent_loop",
        {
            "agent_loop_iterations": 2,
            "tool_call_count": 1,
            "agent_loop_done": True,
            "agent_loop_exit_reason": "finish",
            "tool_trace": [
                {
                    "name": "kapruka_search_products",
                    "args": {"query": "cakes"},
                    "result": {"results": [huge_product] * 30},
                },
            ],
        },
    )
    assert summary["iterations"] == 2
    assert summary["tool_call_count"] == 1
    assert summary["tool_names"] == ["kapruka_search_products"]
    assert summary["exit_reason"] == "finish"
    assert summary["trace_sizes"] == [{"tool": "kapruka_search_products", "products": 30}]
    assert "tool_trace" not in summary
    assert huge_product["image_url"] not in str(summary)


def test_summarize_node_update_agent_loop_exit_reasons() -> None:
    ask_user = summarize_node_update(
        "agent_loop",
        {
            "agent_clarifying_question": "Which city?",
            "agent_loop_iterations": 1,
            "tool_trace": [],
        },
    )
    assert ask_user["exit_reason"] == "ask_user"

    duplicate = summarize_node_update(
        "agent_loop",
        {
            "agent_loop_exit_reason": "duplicate_guard",
            "agent_loop_iterations": 2,
            "tool_trace": [],
        },
    )
    assert duplicate["exit_reason"] == "duplicate_guard"

    max_iter = summarize_node_update(
        "agent_loop",
        {
            "agent_loop_exit_reason": "max_iterations",
            "agent_loop_iterations": 4,
            "tool_trace": [],
        },
    )
    assert max_iter["exit_reason"] == "max_iterations"


def test_trace_agent_iteration_emits_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    configure_dev_logging()
    with caplog.at_level(logging.INFO, logger="agentic.trace"):
        trace_agent_iteration(
            0,
            "kapruka_search_products",
            {"query": "birthday cakes", "limit": 20},
        )
    assert any("AGENT LOOP ▶ iteration 0" in record.message for record in caplog.records)
    assert any("kapruka_search_products" in record.message for record in caplog.records)
    assert any("birthday cakes" in record.message for record in caplog.records)


def test_trace_agent_iteration_omits_full_mcp_payloads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Per-iteration trace must not log full product catalogs or planner rationale."""
    monkeypatch.setenv("APP_ENV", "development")
    configure_dev_logging()
    secret_payload = {"results": [{"id": f"p{i}", "name": f"Product {i}"} for i in range(50)]}
    with caplog.at_level(logging.INFO, logger="agentic.trace"):
        trace_agent_iteration(
            1,
            "kapruka_search_products",
            {"query": "roses", "full_catalog": secret_payload},
        )
    log_text = "\n".join(record.message for record in caplog.records)
    assert "Product 49" not in log_text
    assert "full_catalog" in log_text or "query" in log_text


def test_trace_agent_iteration_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG_TRACE", "0")
    with caplog.at_level(logging.INFO, logger="agentic.trace"):
        trace_agent_iteration(0, "kapruka_search_products", {"query": "cakes"})
    assert not caplog.records
