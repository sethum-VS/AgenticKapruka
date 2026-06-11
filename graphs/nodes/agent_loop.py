"""Bounded ReAct agent loop node — planner trace summarization and prompt helpers."""

from __future__ import annotations

import json
from typing import Any, TypedDict
from urllib.parse import urlparse

from graphs.state import ToolInvocation
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL

PLANNER_SEARCH_RESULT_LIMIT = 5
PLANNER_CATEGORY_NODE_LIMIT = 10

_SEARCH_PRODUCT_FIELDS = frozenset({"id", "name", "price", "in_stock"})
_GET_PRODUCT_FIELDS = frozenset({"id", "name", "price", "in_stock"})


class PlannerTraceEntry(TypedDict):
    """Planner-safe view of a prior tool invocation — summary only, never full result."""

    name: str
    args: dict[str, Any]
    summary: Any


def _summarize_error(result: dict[str, Any]) -> dict[str, Any]:
    return {"error": result.get("error"), "message": result.get("message")}


def _category_id_from_url(url: str) -> str:
    """Derive a stable category id from a Kapruka category URL path segment."""
    path = urlparse(url).path.rstrip("/")
    if not path:
        return url
    return path.rsplit("/", 1)[-1] or url


def _flatten_category_nodes(
    nodes: list[dict[str, Any]],
    *,
    limit: int = PLANNER_CATEGORY_NODE_LIMIT,
) -> list[dict[str, str]]:
    """Walk category tree depth-first, collecting name + id pairs up to limit."""
    collected: list[dict[str, str]] = []

    def walk(items: list[dict[str, Any]]) -> None:
        for node in items:
            if len(collected) >= limit:
                return
            if not isinstance(node, dict):
                continue
            name = node.get("name")
            if isinstance(name, str) and name:
                url = node.get("url")
                category_id = _category_id_from_url(url) if isinstance(url, str) and url else name
                collected.append({"name": name, "id": category_id})
            children = node.get("children")
            if isinstance(children, list) and children:
                walk(children)

    walk(nodes)
    return collected


def _summarize_search_products(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return _summarize_error(result)
    raw_results = result.get("results")
    if not isinstance(raw_results, list):
        return {"results": []}
    summarized: list[dict[str, Any]] = []
    for product in raw_results[:PLANNER_SEARCH_RESULT_LIMIT]:
        if not isinstance(product, dict):
            continue
        summarized.append({key: product[key] for key in _SEARCH_PRODUCT_FIELDS if key in product})
    return {"results": summarized}


def _summarize_get_product(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return _summarize_error(result)
    return {key: result[key] for key in _GET_PRODUCT_FIELDS if key in result}


def _summarize_list_categories(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return _summarize_error(result)
    categories = result.get("categories")
    if not isinstance(categories, list):
        return {"categories": []}
    return {"categories": _flatten_category_nodes(categories)}


def _summarize_check_delivery(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return _summarize_error(result)
    return {
        "city": result.get("city"),
        "deliverable": result.get("available"),
    }


def _summarize_for_planner(name: str, result: Any) -> Any:
    """Return a compact tool payload safe for planner prompt context.

    Full MCP results remain in ``tool_trace[].result`` for ``generate_response``.
    """
    if isinstance(result, dict) and "error" in result:
        return _summarize_error(result)
    if name == SEARCH_PRODUCTS_TOOL:
        return _summarize_search_products(result)
    if name == GET_PRODUCT_TOOL:
        return _summarize_get_product(result)
    if name == LIST_CATEGORIES_TOOL:
        return _summarize_list_categories(result)
    if name == CHECK_DELIVERY_TOOL:
        return _summarize_check_delivery(result)
    if isinstance(result, dict):
        return _summarize_error(result) if "error" in result else {"status": "ok"}
    return result


def build_planner_prior_iterations(
    tool_trace: list[ToolInvocation] | None,
) -> list[PlannerTraceEntry]:
    """Build planner-safe prior-iteration context — never includes full MCP payloads."""
    if not tool_trace:
        return []
    return [
        {
            "name": invocation["name"],
            "args": invocation["args"],
            "summary": _summarize_for_planner(invocation["name"], invocation["result"]),
        }
        for invocation in tool_trace
    ]


def format_planner_prior_iterations(tool_trace: list[ToolInvocation] | None) -> str:
    """Serialize prior tool iterations for injection into the planner system prompt."""
    entries = build_planner_prior_iterations(tool_trace)
    if not entries:
        return ""
    return json.dumps(entries, ensure_ascii=False)
