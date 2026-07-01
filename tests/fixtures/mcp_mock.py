"""Deterministic Kapruka MCP JSON responses for CI and Ragas evaluation."""

from __future__ import annotations

import copy
import json
from typing import Any

from lib.kapruka.tools.create_order import TOOL_NAME as CREATE_ORDER_TOOL
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL

SEARCH_PRODUCTS_JSON: dict[str, Any] = {
    "results": [
        {
            "id": "cake00ka002034",
            "name": "Chocolate Birthday Cake",
            "summary": "Rich chocolate cake for celebrations.",
            "price": {"amount": 4500.0, "currency": "LKR"},
            "compare_at_price": None,
            "in_stock": True,
            "stock_level": "high",
            "image_url": "https://static2.kapruka.com/product-image/cake.jpg",
            "category": {"id": "cat_cakes", "name": "Birthday", "slug": "birthday"},
            "rating": None,
            "ships_internationally": False,
            "url": "https://www.kapruka.com/buyonline/chocolate-birthday-cake/kid/cake00ka002034",
        },
        {
            "id": "flower00ka001122",
            "name": "Red Rose Bouquet",
            "summary": "Fresh red roses for anniversaries.",
            "price": {"amount": 3200.0, "currency": "LKR"},
            "compare_at_price": None,
            "in_stock": True,
            "stock_level": "medium",
            "image_url": "https://static2.kapruka.com/product-image/roses.jpg",
            "category": {"id": "cat_flowers", "name": "Flowers", "slug": "flowers"},
            "rating": None,
            "ships_internationally": False,
            "url": "https://www.kapruka.com/buyonline/red-roses/kid/flower00ka001122",
        },
    ],
    "next_cursor": None,
    "applied_filters": {"q": "birthday cake", "limit": 10, "in_stock_only": False},
}

GET_PRODUCT_JSON: dict[str, Any] = {
    "id": "cake00ka002034",
    "name": "Chocolate Birthday Cake",
    "description": "Rich chocolate sponge with buttercream frosting.",
    "summary": "Perfect for birthday celebrations.",
    "price": {"amount": 4500.0, "currency": "LKR"},
    "compare_at_price": None,
    "in_stock": True,
    "stock_level": "high",
    "category": {
        "id": "cat_cakes",
        "name": "Birthday",
        "slug": "birthday",
        "path": "Cakes > Birthday",
    },
    "variants": [],
    "images": ["https://static2.kapruka.com/product-image/cake.jpg"],
    "attributes": {
        "type": "cake",
        "subtype": "birthday",
        "weight": "1kg",
        "vendor": "Kapruka Bakery",
    },
    "shipping": {
        "ships_from": "Colombo",
        "ships_internationally": False,
        "restricted_countries": [],
    },
    "rating": None,
    "url": "https://www.kapruka.com/buyonline/chocolate-birthday-cake/kid/cake00ka002034",
}

LIST_CATEGORIES_JSON: dict[str, Any] = {
    "categories": [
        {
            "name": "Cakes",
            "url": "https://www.kapruka.com/online/cakes",
            "children": [
                {
                    "name": "Birthday",
                    "url": "https://www.kapruka.com/online/cakes/birthday",
                    "children": [],
                },
            ],
        },
        {
            "name": "Flowers",
            "url": "https://www.kapruka.com/online/flowers",
            "children": [],
        },
        {
            "name": "Chocolates",
            "url": "https://www.kapruka.com/online/chocolates",
            "children": [],
        },
    ],
}

LIST_DELIVERY_CITIES_JSON: dict[str, Any] = {
    "cities": [
        {"name": "Colombo 03", "aliases": ["Colombo 3", "Kollupitiya"]},
        {"name": "Colombo 07", "aliases": ["Cinnamon Gardens"]},
        {"name": "Galle", "aliases": []},
        {"name": "Kandy", "aliases": []},
    ],
    "total_matched": 4,
    "showing": 4,
}

CHECK_DELIVERY_AVAILABLE_JSON: dict[str, Any] = {
    "city": "Colombo 03",
    "now": "2026-06-07T10:30:00+05:30",
    "checked_date": "2026-06-08",
    "available": True,
    "rate": 350.0,
    "currency": "LKR",
    "reason": None,
    "next_available_date": None,
    "perishable_warning": None,
}

CREATE_ORDER_JSON: dict[str, Any] = {
    "checkout_url": "https://www.kapruka.com/checkout/pay/eval-mock-001",
    "order_ref": "ORD-20260608-9901",
    "summary": {
        "items_total": 4500.0,
        "delivery_fee": 350.0,
        "addons_total": 0.0,
        "grand_total": 4850.0,
        "currency": "LKR",
    },
    "expires_at": "2026-06-08T12:30:00+05:30",
}

TRACK_ORDER_JSON: dict[str, Any] = {
    "order_number": "VIMP34456CB2",
    "pnref": "12345678901",
    "status": "shipped",
    "status_display": "Out for Delivery",
    "order_date": "June 5, 2026",
    "delivery_date": "June 7, 2026",
    "shipped_date": "June 6, 2026",
    "amount": "15500.00",
    "payment_method": "Visa",
    "comments": None,
    "recipient": {
        "name": "Ada Lovelace",
        "phone": "0771234567",
        "address": "123 Galle Road",
        "city": "Colombo 03",
    },
    "greeting_message": None,
    "special_instructions": None,
    "progress": [
        {"step": "received", "timestamp": "June 5, 2026 10:00 AM"},
        {"step": "shipped", "timestamp": "June 6, 2026 08:00 AM"},
    ],
    "live_tracking_available": False,
    "has_delivery_video": False,
    "has_delivery_photo": False,
    "items": [
        {
            "product_id": "cake00ka002034",
            "name": "Chocolate Fudge Cake",
            "quantity": 1,
            "selling_price": 4500.0,
        },
    ],
}

ALL_MOCK_TOOL_NAMES: frozenset[str] = frozenset(
    {
        SEARCH_PRODUCTS_TOOL,
        GET_PRODUCT_TOOL,
        LIST_CATEGORIES_TOOL,
        LIST_CITIES_TOOL,
        CHECK_DELIVERY_TOOL,
        CREATE_ORDER_TOOL,
        TRACK_ORDER_TOOL,
    },
)


def _search_payload(params: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(SEARCH_PRODUCTS_JSON)
    filters = dict(payload["applied_filters"])
    if params.get("q"):
        filters["q"] = params["q"]
    if params.get("category"):
        filters["category"] = params["category"]
    if params.get("in_stock_only") is not None:
        filters["in_stock_only"] = params["in_stock_only"]
    payload["applied_filters"] = filters
    return payload


def _get_product_payload(params: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(GET_PRODUCT_JSON)
    product_id = params.get("product_id")
    if isinstance(product_id, str) and product_id:
        payload["id"] = product_id
    return payload


def _check_delivery_payload(params: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(CHECK_DELIVERY_AVAILABLE_JSON)
    city = params.get("city")
    if isinstance(city, str) and city:
        payload["city"] = city
    delivery_date = params.get("delivery_date")
    if isinstance(delivery_date, str) and delivery_date:
        payload["checked_date"] = delivery_date
    product_id = params.get("product_id")
    if isinstance(product_id, str) and product_id.upper().startswith(
        ("CAKE", "FLOWER", "COMBO", "CHOC"),
    ):
        payload["perishable_warning"] = (
            "Perishable items are freshest when delivered within one day."
        )
    return payload


def _track_order_payload(params: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(TRACK_ORDER_JSON)
    order_number = params.get("order_number")
    if isinstance(order_number, str) and order_number:
        payload["order_number"] = order_number
    return payload


def mock_mcp_response(tool_name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return deterministic JSON payload for a Kapruka MCP tool call."""
    args = params or {}
    if tool_name == SEARCH_PRODUCTS_TOOL:
        return _search_payload(args)
    if tool_name == GET_PRODUCT_TOOL:
        return _get_product_payload(args)
    if tool_name == LIST_CATEGORIES_TOOL:
        return copy.deepcopy(LIST_CATEGORIES_JSON)
    if tool_name == LIST_CITIES_TOOL:
        return copy.deepcopy(LIST_DELIVERY_CITIES_JSON)
    if tool_name == CHECK_DELIVERY_TOOL:
        return _check_delivery_payload(args)
    if tool_name == CREATE_ORDER_TOOL:
        return copy.deepcopy(CREATE_ORDER_JSON)
    if tool_name == TRACK_ORDER_TOOL:
        return _track_order_payload(args)
    msg = f"Unexpected MCP tool in mock fixture: {tool_name}"
    raise ValueError(msg)


class MockMCPHttpClient:
    """In-memory MCP transport returning deterministic JSON for all 7 Kapruka tools."""

    def __init__(self, url: str = "mock://kapruka/mcp") -> None:
        self._url = url
        self.call_log: list[str] = []

    @classmethod
    async def connect(
        cls,
        url: str = "mock://kapruka/mcp",
        *,
        httpx_client: Any | None = None,
    ) -> MockMCPHttpClient:
        """Match MCPHttpClient.connect signature for KaprukaService wiring."""
        _ = httpx_client
        return cls(url)

    async def call_tool(self, name: str, params: dict[str, Any] | None = None) -> str:
        """Return JSON text for the requested Kapruka MCP tool."""
        self.call_log.append(name)
        payload = mock_mcp_response(name, params)
        return json.dumps(payload, ensure_ascii=False)

    async def close(self) -> None:
        """No-op close for interface parity with MCPHttpClient."""
