"""Shared Kapruka MCP tool dispatcher with validation, currency, and serialization."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from graphs.nodes.resolve_cart_product import match_products_by_phrase
from lib.chat.product_reference import (
    _normalize_ordinal_phrase,
    is_ordinal_phrase,
    resolve_product_reference,
)
from graphs.state import AgentState
from lib.kapruka.errors import KaprukaError, KaprukaRateLimitError
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import (
    CheckDeliveryInput,
    DeliveryCity,
    GetProductInput,
    ListCategoriesInput,
    ListDeliveryCitiesInput,
    ListDeliveryCitiesOutput,
    SearchProductsInput,
    TrackOrderInput,
)
from lib.redis.rate_limit import RateLimitExceeded

logger = logging.getLogger(__name__)

SUPPORTED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        SEARCH_PRODUCTS_TOOL,
        GET_PRODUCT_TOOL,
        LIST_CATEGORIES_TOOL,
        TRACK_ORDER_TOOL,
        CHECK_DELIVERY_TOOL,
        LIST_CITIES_TOOL,
    },
)

_CURRENCY_TOOLS: frozenset[str] = frozenset({SEARCH_PRODUCTS_TOOL, GET_PRODUCT_TOOL})

_TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    SEARCH_PRODUCTS_TOOL: SearchProductsInput,
    GET_PRODUCT_TOOL: GetProductInput,
    LIST_CATEGORIES_TOOL: ListCategoriesInput,
    TRACK_ORDER_TOOL: TrackOrderInput,
    CHECK_DELIVERY_TOOL: CheckDeliveryInput,
    LIST_CITIES_TOOL: ListDeliveryCitiesInput,
}

_SERVICE_KWARG_EXCLUDE: dict[str, frozenset[str]] = {
    SEARCH_PRODUCTS_TOOL: frozenset({"response_format", "include_stubs"}),
    GET_PRODUCT_TOOL: frozenset({"response_format"}),
    LIST_CATEGORIES_TOOL: frozenset({"response_format"}),
    TRACK_ORDER_TOOL: frozenset({"response_format"}),
    CHECK_DELIVERY_TOOL: frozenset({"response_format"}),
    LIST_CITIES_TOOL: frozenset({"response_format"}),
}


def normalize_planner_tool_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Coerce common planner aliases into Kapruka tool input field names."""
    normalized = dict(args)
    if name == SEARCH_PRODUCTS_TOOL:
        query = normalized.get("query")
        if "q" not in normalized and isinstance(query, str) and query.strip():
            normalized["q"] = query.strip()
            normalized.pop("query", None)
        category_id = normalized.get("category_id")
        if "category" not in normalized and isinstance(category_id, str) and category_id.strip():
            normalized["category"] = category_id.strip()
            normalized.pop("category_id", None)
    if name == CHECK_DELIVERY_TOOL:
        city = normalized.get("city")
        for alias in ("delivery_city", "city_name", "destination"):
            alias_value = normalized.get(alias)
            if (
                (not isinstance(city, str) or not city.strip())
                and isinstance(alias_value, str)
                and alias_value.strip()
            ):
                normalized["city"] = alias_value.strip()
                city = normalized["city"]
            normalized.pop(alias, None)
        normalized.pop("q", None)
        delivery_date = normalized.get("delivery_date")
        date_alias = normalized.get("date")
        if (
            (not isinstance(delivery_date, str) or not delivery_date.strip())
            and isinstance(date_alias, str)
            and date_alias.strip()
        ):
            normalized["delivery_date"] = date_alias.strip()
            normalized.pop("date", None)
    return normalized


def enrich_get_product_args(
    args: dict[str, Any],
    state: AgentState,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve product_id from carousel/session context before kapruka_get_product."""
    enriched = dict(args)
    product_id = enriched.get("product_id")
    if isinstance(product_id, str) and product_id.strip():
        return enriched, None

    last_search = [
        item for item in (state.get("last_search_products") or []) if isinstance(item, dict)
    ]
    name_keys = ("q", "product_name", "name")
    phrase = ""
    for key in name_keys:
        raw = enriched.get(key)
        if isinstance(raw, str) and raw.strip():
            phrase = raw.strip()
            break

    if not phrase:
        session_focus = state.get("session_product_focus")
        if isinstance(session_focus, str) and session_focus.strip():
            phrase = session_focus.strip()

    last_visible = [
        item for item in (state.get("last_visible_products") or []) if isinstance(item, dict)
    ]

    if phrase:
        ordinal_phrase = _normalize_ordinal_phrase(phrase)
        if is_ordinal_phrase(ordinal_phrase):
            reference = resolve_product_reference(
                ordinal_phrase,
                last_visible_products=last_visible or None,
                last_search_products=last_search or None,
                session_product_focus=state.get("session_product_focus"),
            )
            if reference is not None and reference.get("status") == "resolved":
                product = reference.get("product")
                if isinstance(product, dict) and product.get("id") is not None:
                    resolved = {**enriched, "product_id": str(product["id"])}
                    for key in name_keys:
                        resolved.pop(key, None)
                    return resolved, None

    if phrase and last_search:
        for candidates in (last_visible, last_search):
            if not candidates:
                continue
            for threshold in (0.6, 0.4):
                matched, _tied, _question = match_products_by_phrase(
                    phrase,
                    candidates,
                    threshold=threshold,
                )
                if matched is not None:
                    pid = matched.get("id")
                    if pid is not None:
                        resolved = {**enriched, "product_id": str(pid)}
                        for key in name_keys:
                            resolved.pop(key, None)
                        return resolved, None

    if phrase or any(key in args for key in name_keys):
        return enriched, {
            "error": "product_id_unresolved",
            "message": "Could not resolve product name to a Kapruka product id.",
        }

    return enriched, None


def canonical_tool_args_for_dedup(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize args for duplicate-tool detection (ignore session currency injection)."""
    canonical = normalize_planner_tool_args(name, dict(args))
    if name in _CURRENCY_TOOLS:
        canonical.pop("currency", None)
    return canonical


def inject_currency(
    name: str,
    args: dict[str, Any],
    currency: str,
    *,
    budget_currency: str | None = None,
) -> dict[str, Any]:
    """Ensure price-bearing MCP tools receive session or explicit budget currency."""
    if name not in _CURRENCY_TOOLS or "currency" in args:
        return args
    effective = budget_currency if budget_currency else currency
    return {**args, "currency": effective}


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
    if name == LIST_CITIES_TOOL:
        names = await service.list_delivery_cities(client_ip, **kwargs)
        return ListDeliveryCitiesOutput(
            cities=[DeliveryCity(name=name) for name in names],
            total_matched=len(names),
            showing=len(names),
        )
    msg = f"Unsupported MCP tool: {name}"
    raise ValueError(msg)


async def invoke_tool(
    name: str,
    args: dict[str, Any],
    *,
    kapruka_service: KaprukaService,
    client_ip: str,
    currency: str = "LKR",
    budget_currency: str | None = None,
) -> dict[str, Any]:
    """Validate args, invoke KaprukaService, and return a serialized MCP payload.

    On Kapruka MCP or local validation failure, returns ``{"error": code, "message": ...}``.
    """
    enriched = inject_currency(
        name,
        normalize_planner_tool_args(name, dict(args)),
        currency,
        budget_currency=budget_currency,
    )

    try:
        validated = _validate_tool_args(name, enriched)
    except ValidationError as exc:
        return {"error": "validation_error", "message": _format_validation_error(exc)}
    except ValueError as exc:
        return {"error": "unsupported_tool", "message": str(exc)}

    try:
        raw = await _dispatch_tool(kapruka_service, client_ip, name, validated)
    except KaprukaRateLimitError as exc:
        logger.warning("invoke_tool: %s rate limited", name, exc_info=True)
        return {
            "error": "rate_limit_exceeded",
            "message": exc.message,
            "retry_after_seconds": exc.retry_after_seconds,
        }
    except RateLimitExceeded as exc:
        logger.warning("invoke_tool: %s app rate limit", name, exc_info=True)
        return {
            "error": "rate_limit_exceeded",
            "message": str(exc),
            "retry_after_seconds": exc.retry_after_seconds,
        }
    except KaprukaError as exc:
        from app.middleware.errors import human_readable_message

        logger.warning("invoke_tool: %s failed (%s)", name, exc.code, exc_info=True)
        return {"error": exc.code, "message": human_readable_message(exc)}

    return serialize_tool_result(raw)
