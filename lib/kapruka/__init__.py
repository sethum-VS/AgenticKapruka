"""Kapruka MCP client utilities."""

from lib.kapruka.errors import (
    KaprukaError,
    KaprukaNotFoundError,
    KaprukaRateLimitError,
    KaprukaValidationError,
    parse_mcp_error,
)
from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.tools.get_product import get_product
from lib.kapruka.tools.list_categories import list_categories
from lib.kapruka.tools.search_products import search_products
from lib.kapruka.types import (
    CartItem,
    CreateOrderInput,
    CreateOrderResponse,
    Delivery,
    GetProductInput,
    GetProductOutput,
    ListCategoriesInput,
    ListCategoriesOutput,
    Recipient,
    SearchProductsInput,
    SearchProductsOutput,
    Sender,
    TrackOrderInput,
    TrackOrderOutput,
)

__all__ = [
    "KaprukaError",
    "KaprukaNotFoundError",
    "KaprukaRateLimitError",
    "KaprukaValidationError",
    "MCPHttpClient",
    "parse_mcp_error",
    "CartItem",
    "CreateOrderInput",
    "CreateOrderResponse",
    "Delivery",
    "GetProductInput",
    "GetProductOutput",
    "ListCategoriesInput",
    "ListCategoriesOutput",
    "Recipient",
    "SearchProductsInput",
    "SearchProductsOutput",
    "Sender",
    "get_product",
    "list_categories",
    "search_products",
    "TrackOrderInput",
    "TrackOrderOutput",
]
