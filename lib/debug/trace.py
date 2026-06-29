"""Structured console tracing for local chat debugging."""

from __future__ import annotations

import json
import logging
import os
import traceback
from typing import Any

logger = logging.getLogger("agentic.trace")

# Third-party libraries that spam DEBUG/INFO on every HTTP retry or MCP SSE frame.
_QUIET_THIRD_PARTY_LOGGERS = (
    "asyncio",
    "google.auth",
    "google_genai",
    "httpcore",
    "httpx",
    "mcp",
    "urllib3",
    "uvicorn.access",
    "watchfiles",
)

_TRACE_TRUE = frozenset({"1", "true", "yes", "on"})
_TRACE_FALSE = frozenset({"0", "false", "no", "off"})
_MAX_STRING = 400
_MAX_LIST_ITEMS = 8
_SKIP_KEYS = frozenset({"embedding", "embeddings", "vector"})


def is_debug_trace_enabled() -> bool:
    """True when verbose pipeline tracing should print to the console."""
    explicit = os.getenv("DEBUG_TRACE")
    if explicit is not None:
        normalized = explicit.strip().lower()
        if normalized in _TRACE_FALSE:
            return False
        if normalized in _TRACE_TRUE:
            return True
    return os.getenv("APP_ENV", "development").lower() != "production"


def _silence_noisy_loggers() -> None:
    """Keep pipeline trace + app logs readable; hide HTTP/SDK chatter."""
    for name in _QUIET_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_dev_logging() -> None:
    """Configure root logging for local development (idempotent).

    ``DEBUG_TRACE`` controls structured ``agentic.trace`` blocks (CHAT TURN,
    NODE, ROUTE). ``LOG_LEVEL`` controls application log verbosity; default
    INFO keeps ``make logs`` readable while still showing graph node summaries.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        root.setLevel(level)

    logger.setLevel(logging.INFO)
    _silence_noisy_loggers()


def _truncate(value: str, *, limit: int = _MAX_STRING) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}… (+{len(value) - limit} chars)"


def summarize_value(value: Any, *, depth: int = 0) -> Any:
    """Return a log-safe, truncated representation of arbitrary values."""
    if depth > 4:
        return "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, dict):
        return {
            str(key): summarize_value(
                val,
                depth=depth + 1,
            )
            for key, val in value.items()
            if str(key).lower() not in _SKIP_KEYS
        }
    if isinstance(value, (list, tuple)):
        items = [summarize_value(item, depth=depth + 1) for item in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            items.append(f"… +{len(value) - _MAX_LIST_ITEMS} more")
        return items
    return _truncate(repr(value))


def _summarize_tool_trace_sizes(tool_trace: list[Any]) -> list[dict[str, Any]]:
    """Compact per-invocation result sizes — never full MCP payloads."""
    sizes: list[dict[str, Any]] = []
    for invocation in tool_trace:
        if not isinstance(invocation, dict):
            continue
        name = invocation.get("name", "?")
        result = invocation.get("result")
        entry: dict[str, Any] = {"tool": name}
        if isinstance(result, dict):
            if "error" in result:
                entry["error"] = result.get("error")
            elif isinstance(result.get("results"), list):
                entry["products"] = len(result["results"])
            elif isinstance(result.get("categories"), list):
                entry["categories"] = len(result["categories"])
            elif "available" in result or "city" in result:
                entry["delivery"] = True
            else:
                entry["keys"] = len(result)
        elif result is not None:
            entry["type"] = type(result).__name__
        sizes.append(entry)
    return sizes


def _resolve_agent_loop_exit_reason(update: dict[str, Any]) -> str:
    """Resolve agent_loop exit reason from explicit field or state hints."""
    explicit = update.get("agent_loop_exit_reason")
    if isinstance(explicit, str) and explicit:
        return explicit
    if update.get("agent_clarifying_question"):
        return "ask_user"
    if update.get("agent_tool_error"):
        return "tool_error"
    return "finish"


def summarize_node_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    """Shape a LangGraph node delta for readable console output."""
    if node_name == "agent_loop":
        tool_trace = update.get("tool_trace") or []
        tool_names = [
            str(inv["name"]) for inv in tool_trace if isinstance(inv, dict) and inv.get("name")
        ]
        return {
            "iterations": update.get("agent_loop_iterations"),
            "tool_call_count": update.get("tool_call_count"),
            "tool_names": tool_names,
            "exit_reason": _resolve_agent_loop_exit_reason(update),
            "trace_sizes": _summarize_tool_trace_sizes(tool_trace),
            "agent_loop_done": update.get("agent_loop_done"),
        }
    if node_name == "generate_response":
        assistant = (update.get("assistant_message") or "").strip()
        html_len = len(update.get("response_html") or "")
        return {
            "assistant_message": _truncate(assistant),
            "response_html_chars": html_len,
        }
    if node_name == "call_mcp_tools":
        results = update.get("tool_results") or {}
        summary: dict[str, Any] = {"tool_call_count": update.get("tool_call_count")}
        tool_summary: dict[str, Any] = {}
        for tool_name, payload in results.items():
            if isinstance(payload, dict) and "error" in payload:
                tool_summary[tool_name] = {
                    "error": payload.get("error"),
                    "message": payload.get("message"),
                }
            elif isinstance(payload, dict):
                products = payload.get("products")
                if isinstance(products, list):
                    tool_summary[tool_name] = {
                        "products": len(products),
                        "sample": [
                            p.get("name") or p.get("id")
                            for p in products[:3]
                            if isinstance(p, dict)
                        ],
                    }
                else:
                    tool_summary[tool_name] = summarize_value(payload)
            else:
                tool_summary[tool_name] = summarize_value(payload)
        summary["tool_results"] = tool_summary
        return summary
    if node_name == "retrieve_hybrid_context":
        hybrid = update.get("hybrid_context") or {}
        hints = hybrid.get("hints") or {}
        products = hybrid.get("products") or []
        return {
            "hints": summarize_value(hints),
            "product_count": len(products) if isinstance(products, list) else 0,
            "preferences": summarize_value(hybrid.get("preferences")),
        }
    if node_name == "analyze_intent":
        return {
            "intent": update.get("intent"),
            "intent_metadata": summarize_value(update.get("intent_metadata")),
            "model_tier": update.get("model_tier"),
            "tool_calls": summarize_value(update.get("tool_calls")),
            "specificity_score": update.get("specificity_score"),
            "specificity_band": update.get("specificity_band"),
            "session_awaiting_clarification_dimension": update.get(
                "session_awaiting_clarification_dimension",
            ),
        }
    if node_name == "load_zep_memory":
        facts = update.get("zep_memory_facts") or []
        return {
            "fact_count": len(facts) if isinstance(facts, list) else 0,
            "facts": summarize_value(facts),
        }
    if node_name == "run_checkout_graph":
        return {
            "checkout_state": update.get("checkout_state"),
            "checkout_paused": update.get("checkout_paused"),
            "tool_results": summarize_value(update.get("tool_results")),
        }
    if node_name == "master_flow":
        return {
            "master_flow_invoked": update.get("master_flow_invoked"),
            "master_flow_decision": update.get("master_flow_decision"),
            "master_flow_skip_reason": update.get("master_flow_skip_reason"),
            "master_clarifying_question": summarize_value(
                update.get("master_clarifying_question"),
            ),
            "active_flow": update.get("active_flow"),
            "master_flow_mismatch_reason": summarize_value(
                update.get("master_flow_mismatch_reason"),
            ),
        }
    if node_name == "zep_memory_write":
        return {"persisted": True}
    summarized = summarize_value(update)
    if isinstance(summarized, dict):
        return summarized
    return {"value": summarized}


def _emit_block(title: str, body: str) -> None:
    border = "─" * 72
    logger.info("%s\n%s\n%s", border, title, body)


def trace_turn_start(
    *,
    thread_id: str,
    message: str,
    currency: str | None = None,
    client_ip: str | None = None,
    state: dict[str, Any] | None = None,
) -> None:
    """Log inbound chat request and seeded graph state."""
    if not is_debug_trace_enabled():
        return
    lines = [
        f"thread_id: {thread_id}",
        f"message: {message!r}",
    ]
    if currency:
        lines.append(f"currency: {currency}")
    if client_ip:
        lines.append(f"client_ip: {client_ip}")
    if state:
        lines.append(f"state_seed: {json.dumps(summarize_value(state), ensure_ascii=False)}")
    _emit_block("CHAT TURN ▶ START", "\n".join(lines))


def trace_node_update(node_name: str, update: dict[str, Any]) -> None:
    """Log a LangGraph node output delta."""
    if not is_debug_trace_enabled():
        return
    summary = summarize_node_update(node_name, update)
    body = json.dumps(summary, ensure_ascii=False, indent=2)
    _emit_block(f"NODE ▶ {node_name}", body)


def trace_agent_iteration(
    iteration: int,
    tool_name: str | None,
    args_summary: dict[str, Any],
) -> None:
    """Log one agent-loop planner iteration without full MCP payloads."""
    if not is_debug_trace_enabled():
        return
    payload = {
        "iteration": iteration,
        "tool_name": tool_name,
        "args": summarize_value(args_summary),
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    _emit_block(f"AGENT LOOP ▶ iteration {iteration}", body)


def trace_route_decision(
    *,
    from_node: str,
    target: str,
    intent: str | None = None,
    reason: str | None = None,
) -> None:
    """Log conditional graph routing."""
    if not is_debug_trace_enabled():
        return
    lines = [f"{from_node} → {target}"]
    if intent:
        lines.append(f"intent: {intent}")
    if reason:
        lines.append(f"reason: {reason}")
    _emit_block("ROUTE", "\n".join(lines))


def trace_master_flow(
    *,
    skipped: bool,
    skip_reason: str | None = None,
    trigger_reason: str | None = None,
    active_flow: str | None = None,
    decision: str | None = None,
    confidence: float | None = None,
    mismatch_reason: str | None = None,
    patches_applied: bool | None = None,
) -> None:
    """Log master flow supervisor invocation, skip, or patch outcome."""
    if skipped:
        logger.info("master_flow_skipped: %s", skip_reason or "no_trigger")
    if not is_debug_trace_enabled():
        return
    lines: list[str] = []
    if skipped:
        lines.append(f"skipped: {skip_reason or 'no_trigger'}")
    else:
        lines.append("invoked: true")
        if trigger_reason:
            lines.append(f"trigger: {trigger_reason}")
    if active_flow:
        lines.append(f"active_flow: {active_flow}")
    if decision:
        lines.append(f"decision: {decision}")
    if confidence is not None:
        lines.append(f"confidence: {confidence:.2f}")
    if mismatch_reason:
        lines.append(f"mismatch: {mismatch_reason}")
    if patches_applied is not None:
        lines.append(f"patches_applied: {patches_applied}")
    _emit_block("MASTER FLOW", "\n".join(lines))


def trace_turn_complete(
    *,
    thread_id: str,
    assistant_message: str | None = None,
    response_html_chars: int | None = None,
) -> None:
    """Log final assistant output for a completed turn."""
    if not is_debug_trace_enabled():
        return
    lines = [f"thread_id: {thread_id}"]
    if assistant_message:
        lines.append(f"assistant_message: {_truncate(assistant_message.strip())}")
    if response_html_chars is not None:
        lines.append(f"response_html_chars: {response_html_chars}")
    _emit_block("CHAT TURN ▶ COMPLETE", "\n".join(lines))


def trace_error(context: str, exc: BaseException | None = None) -> None:
    """Log a pipeline error with optional traceback."""
    if not is_debug_trace_enabled():
        return
    lines = [f"context: {context}"]
    if exc is not None:
        lines.append(f"error: {type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())
    _emit_block("CHAT TURN ▶ ERROR", "\n".join(lines))
