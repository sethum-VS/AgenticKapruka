"""Shared Kapruka MCP tool dispatcher with validation, currency, and serialization."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from lib.kapruka.errors import KaprukaError
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import (
    CheckDeliveryInput,
    GetProductInput,
    ListCategoriesInput,
    SearchProductsInput,
    TrackOrderInput,
)

logger = logging.getLogger(__name__)

SUPPORTED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        SEARCH_PRODUCTS_TOOL,
        GET_PRODUCT_TOOL,
        LIST_CATEGORIES_TOOL,
        TRACK_ORDER_TOOL,
        CHECK_DELIVERY_TOOL,
    },
)

_CURRENCY_TOOLS: frozenset[str] = frozenset({SEARCH_PRODUCTS_TOOL, GET_PRODUCT_TOOL})

_TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    SEARCH_PRODUCTS_TOOL: SearchProductsInput,
    GET_PRODUCT_TOOL: GetProductInput,
    LIST_CATEGORIES_TOOL: ListCategoriesInput,
    TRACK_ORDER_TOOL: TrackOrderInput,
    CHECK_DELIVERY_TOOL: CheckDeliveryInput,
}

_SERVICE_KWARG_EXCLUDE: dict[str, frozenset[str]] = {
    SEARCH_PRODUCTS_TOOL: frozenset({"response_format", "include_stubs"}),
    GET_PRODUCT_TOOL: frozenset({"response_format"}),
    LIST_CATEGORIES_TOOL: frozenset({"response_format"}),
    TRACK_ORDER_TOOL: frozenset({"response_format"}),
    CHECK_DELIVERY_TOOL: frozenset({"response_format"}),
}


def inject_currency(name: str, args: dict[str, Any], currency: str) -> dict[str, Any]:
    """Ensure price-bearing MCP tools receive the session currency."""
    if name in _CURRENCY_TOOLS and "currency" not in args:
        return {**args, "currency": currency}
    return args


def serialize_tool_result(result: Any) -> dict[str, Any]:
    """Convert Pydantic tool outputs to JSON-serializable dicts."""
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, dict):
        return result
    msg = f"Cannot serialize tool result of type {type(result).__name__}"
    raise TypeError(msg)


def _format_validation_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first.get("loc", ()))
    msg = str(first.get("msg", "Invalid input"))
    if loc:
        return f"{loc}: {msg}"
    return msg


def _validate_tool_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate tool args against lib/kapruka/types.py models."""
    model_cls = _TOOL_INPUT_MODELS.get(name)
    if model_cls is None:
        msg = f"Unsupported MCP tool: {name}"
        raise ValueError(msg)
    validated = model_cls.model_validate(args)
    return validated.model_dump(mode="json", exclude_unset=True, exclude_none=True)


def _service_kwargs(name: str, validated: dict[str, Any]) -> dict[str, Any]:
    """Strip fields KaprukaService methods do not accept."""
    exclude = _SERVICE_KWARG_EXCLUDE.get(name, frozenset({"response_format"}))
    return {key: value for key, value in validated.items() if key not in exclude}


async def _dispatch_tool(
    service: KaprukaService,
    client_ip: str,
    name: str,
    validated: dict[str, Any],
) -> Any:
    """Invoke KaprukaService after args are validated (cache + rate limits apply)."""
    kwargs = _service_kwargs(name, validated)
    if name == SEARCH_PRODUCTS_TOOL:
        return await service.search_products(client_ip, **kwargs)
    if name == GET_PRODUCT_TOOL:
        return await service.get_product(client_ip, **kwargs)
    if name == LIST_CATEGORIES_TOOL:
        return await service.list_categories(client_ip, **kwargs)
    if name == TRACK_ORDER_TOOL:
        return await service.track_order(client_ip, **kwargs)
    if name == CHECK_DELIVERY_TOOL:
        return await service.check_delivery(client_ip, **kwargs)
    msg = f"Unsupported MCP tool: {name}"
    raise ValueError(msg)


async def invoke_tool(
    name: str,
    args: dict[str, Any],
    *,
    kapruka_service: KaprukaService,
    client_ip: str,
    currency: str = "LKR",
) -> dict[str, Any]:
    """Validate args, invoke KaprukaService, and return a serialized MCP payload.

    On Kapruka MCP or local validation failure, returns ``{"error": code, "message": ...}``.
    """
    enriched = inject_currency(name, dict(args), currency)

    try:
        validated = _validate_tool_args(name, enriched)
    except ValidationError as exc:
        return {"error": "validation_error", "message": _format_validation_error(exc)}
    except ValueError as exc:
        return {"error": "unsupported_tool", "message": str(exc)}

    try:
        raw = await _dispatch_tool(kapruka_service, client_ip, name, validated)
    except KaprukaError as exc:
        from app.middleware.errors import human_readable_message

        logger.warning("invoke_tool: %s failed (%s)", name, exc.code, exc_info=True)
        return {"error": exc.code, "message": human_readable_message(exc)}

    return serialize_tool_result(raw)
