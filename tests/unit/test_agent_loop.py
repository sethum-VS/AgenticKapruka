"""Unit tests for graphs.nodes.agent_loop planner trace summarization."""

from __future__ import annotations

import json

from graphs.nodes.agent_loop import (
    PLANNER_CATEGORY_NODE_LIMIT,
    PLANNER_SEARCH_RESULT_LIMIT,
    build_planner_prior_iterations,
    format_planner_prior_iterations,
)
from graphs.state import ToolInvocation
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL


def _make_search_product(index: int) -> dict[str, object]:
    return {
        "id": f"cake{index:03d}",
        "name": f"Cake {index}",
        "summary": f"Summary for cake {index}",
        "description": f"Long description for cake {index}",
        "price": {"amount": 1000.0 + index, "currency": "LKR"},
        "in_stock": index % 2 == 0,
        "stock_level": "high",
        "image_url": f"https://cdn.kapruka.com/cake{index}.jpg",
        "category": {"id": "cat_cakes", "name": "Cakes", "slug": "cakes", "path": None},
        "variants": [{"id": f"v{index}", "name": "Default"}],
        "images": [f"https://cdn.kapruka.com/cake{index}.jpg"],
        "url": f"https://www.kapruka.com/cake{index}",
    }


def test_summarize_search_products_caps_at_five_and_strips_heavy_fields() -> None:
    """30-product search → planner summary has ≤5 items and no image URLs."""
    full_results = [_make_search_product(i) for i in range(30)]
    full_result_payload = {
        "results": full_results,
        "next_cursor": "page-2",
        "applied_filters": {"q": "birthday cake", "limit": 30},
    }
    tool_trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "birthday cake", "limit": 30},
            "result": full_result_payload,
        }
    ]

    assert len(tool_trace[0]["result"]["results"]) == 30

    planner_entries = build_planner_prior_iterations(tool_trace)
    assert len(planner_entries) == 1
    summary = planner_entries[0]["summary"]
    assert isinstance(summary, dict)
    summarized_results = summary["results"]
    assert isinstance(summarized_results, list)
    assert len(summarized_results) <= PLANNER_SEARCH_RESULT_LIMIT

    planner_blob = format_planner_prior_iterations(tool_trace)
    assert "image_url" not in planner_blob
    assert "Summary for cake" not in planner_blob
    assert "Long description for cake" not in planner_blob
    assert "variants" not in planner_blob

    for product in summarized_results:
        assert set(product.keys()) <= {"id", "name", "price", "in_stock"}
        assert "image_url" not in product


def test_full_tool_trace_retains_all_search_products() -> None:
    """Summarization for planner must not mutate stored tool_trace payloads."""
    full_results = [_make_search_product(i) for i in range(30)]
    invocation: ToolInvocation = {
        "name": SEARCH_PRODUCTS_TOOL,
        "args": {"q": "birthday cake"},
        "result": {"results": full_results, "next_cursor": None, "applied_filters": {}},
    }
    tool_trace = [invocation]

    build_planner_prior_iterations(tool_trace)
    format_planner_prior_iterations(tool_trace)

    assert len(tool_trace[0]["result"]["results"]) == 30
    assert tool_trace[0]["result"]["results"][0]["image_url"].startswith("https://")


def test_summarize_get_product_strips_catalog_heavy_fields() -> None:
    tool_trace: list[ToolInvocation] = [
        {
            "name": GET_PRODUCT_TOOL,
            "args": {"product_id": "cake001"},
            "result": {
                "id": "cake001",
                "name": "Chocolate Cake",
                "description": "Rich chocolate layers",
                "summary": "Rich chocolate",
                "price": {"amount": 4500.0, "currency": "LKR"},
                "in_stock": True,
                "images": ["https://cdn.kapruka.com/cake001.jpg"],
                "variants": [{"id": "v1", "name": "2lb"}],
            },
        }
    ]
    summary = build_planner_prior_iterations(tool_trace)[0]["summary"]
    assert summary == {
        "id": "cake001",
        "name": "Chocolate Cake",
        "price": {"amount": 4500.0, "currency": "LKR"},
        "in_stock": True,
    }
    assert "description" not in json.dumps(summary)


def test_summarize_list_categories_caps_nodes_and_includes_ids() -> None:
    categories = [
        {
            "name": "Cakes",
            "url": "https://www.kapruka.com/shop/cakes",
            "children": [
                {"name": "Birthday", "url": "https://www.kapruka.com/shop/cakes/birthday"},
                {"name": "Wedding", "url": "https://www.kapruka.com/shop/cakes/wedding"},
            ],
        },
        {
            "name": "Flowers",
            "url": "https://www.kapruka.com/shop/flowers",
            "children": [
                {"name": "Roses", "url": "https://www.kapruka.com/shop/flowers/roses"},
            ],
        },
    ]
    tool_trace: list[ToolInvocation] = [
        {
            "name": LIST_CATEGORIES_TOOL,
            "args": {"depth": 2},
            "result": {"categories": categories},
        }
    ]
    summary = build_planner_prior_iterations(tool_trace)[0]["summary"]
    assert isinstance(summary, dict)
    nodes = summary["categories"]
    assert isinstance(nodes, list)
    assert len(nodes) <= PLANNER_CATEGORY_NODE_LIMIT
    assert nodes[0] == {"name": "Cakes", "id": "cakes"}
    assert all("name" in node and "id" in node for node in nodes)


def test_summarize_check_delivery_city_and_deliverable_only() -> None:
    tool_trace: list[ToolInvocation] = [
        {
            "name": CHECK_DELIVERY_TOOL,
            "args": {"city": "Kandy"},
            "result": {
                "city": "Kandy",
                "now": "2026-06-11T10:00:00+05:30",
                "checked_date": "2026-06-12",
                "available": True,
                "rate": 450.0,
                "currency": "LKR",
                "reason": None,
            },
        }
    ]
    summary = build_planner_prior_iterations(tool_trace)[0]["summary"]
    assert summary == {"city": "Kandy", "deliverable": True}
    assert "rate" not in json.dumps(summary)


def test_summarize_errors_message_only() -> None:
    tool_trace: list[ToolInvocation] = [
        {
            "name": SEARCH_PRODUCTS_TOOL,
            "args": {"q": "x"},
            "result": {"error": "product_not_found", "message": "No products found"},
        }
    ]
    summary = build_planner_prior_iterations(tool_trace)[0]["summary"]
    assert summary == {"error": "product_not_found", "message": "No products found"}
