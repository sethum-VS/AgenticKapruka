"""Unified Kapruka MCP facade with per-IP rate limiting and read cache."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from lib.kapruka.mcp_client import MCPHttpClient
from lib.kapruka.tools import (
    check_delivery,
    create_order,
    get_product,
    list_categories,
    list_delivery_cities,
    search_products,
    track_order,
)
from lib.kapruka.tools.create_order import TOOL_NAME as CREATE_ORDER_TOOL
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL, LIST_CITIES_TOOL
from lib.kapruka.tools.get_product import TOOL_NAME as GET_PRODUCT_TOOL
from lib.kapruka.tools.list_categories import TOOL_NAME as LIST_CATEGORIES_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.kapruka.tools.track_order import TOOL_NAME as TRACK_ORDER_TOOL
from lib.kapruka.types import (
    CartItem,
    CheckDeliveryInput,
    CheckDeliveryOutput,
    CreateOrderResponse,
    Delivery,
    DeliveryCity,
    GetProductInput,
    GetProductOutput,
    ListCategoriesInput,
    ListCategoriesOutput,
    ListDeliveryCitiesInput,
    ListDeliveryCitiesOutput,
    Recipient,
    SearchProductsInput,
    SearchProductsOutput,
    Sender,
    TrackOrderInput,
    TrackOrderOutput,
)
from lib.redis.cache import get_cached, set_cached
from lib.redis.client import RedisClient
from lib.redis.rate_limit import check_rate_limit

T = TypeVar("T")


def _cache_args(model: BaseModel) -> dict[str, Any]:
    """Build canonical MCP params dict for cache-key hashing."""
    params = model.model_dump(mode="json", exclude_none=True)
    params["response_format"] = "json"
    return params


class KaprukaService:
    """Facade over Kapruka MCP tools with Redis rate limits and read cache."""

    def __init__(self, redis_client: RedisClient, mcp_client: MCPHttpClient) -> None:
        self._redis = redis_client
        self._mcp = mcp_client

    async def _cached_read(
        self,
        *,
        client_ip: str,
        tool_name: str,
        cache_args: dict[str, Any],
        fetch: Callable[[], Awaitable[T]],
        to_cache: Callable[[T], str],
        from_cache: Callable[[str], T],
    ) -> T:
        """Apply rate limit, return cached read response, or fetch and store."""
        await check_rate_limit(self._redis, client_ip, tool_name)

        cached = await get_cached(self._redis, tool_name, cache_args)
        if cached is not None:
            return from_cache(cached)

        result = await fetch()
        await set_cached(self._redis, tool_name, cache_args, to_cache(result))
        return result

    async def search_products(
        self,
        client_ip: str,
        *,
        q: str,
        category: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        in_stock_only: bool = False,
        sort: str = "relevance",
        limit: int = 10,
        cursor: str | None = None,
        currency: str = "LKR",
    ) -> SearchProductsOutput:
        """Search Kapruka catalog with rate limit and read cache."""
        search_input = SearchProductsInput(
            q=q,
            category=category,
            min_price=min_price,
            max_price=max_price,
            in_stock_only=in_stock_only,
            sort=sort,
            limit=limit,
            cursor=cursor,
            currency=currency,
            response_format="json",
        )
        cache_args = _cache_args(search_input)

        return await self._cached_read(
            client_ip=client_ip,
            tool_name=SEARCH_PRODUCTS_TOOL,
            cache_args=cache_args,
            fetch=lambda: search_products(
                self._mcp,
                q=q,
                category=category,
                min_price=min_price,
                max_price=max_price,
                in_stock_only=in_stock_only,
                sort=sort,
                limit=limit,
                cursor=cursor,
                currency=currency,
            ),
            to_cache=lambda result: result.model_dump_json(),
            from_cache=lambda text: SearchProductsOutput.model_validate(json.loads(text)),
        )

    async def get_product(
        self,
        client_ip: str,
        *,
        product_id: str,
        currency: str = "LKR",
        type: str | None = None,
    ) -> GetProductOutput:
        """Fetch product detail with rate limit and read cache."""
        product_input = GetProductInput(
            product_id=product_id,
            currency=currency,
            type=type,
            response_format="json",
        )
        cache_args = _cache_args(product_input)

        return await self._cached_read(
            client_ip=client_ip,
            tool_name=GET_PRODUCT_TOOL,
            cache_args=cache_args,
            fetch=lambda: get_product(
                self._mcp,
                product_id=product_id,
                currency=currency,
                type=type,
            ),
            to_cache=lambda result: result.model_dump_json(),
            from_cache=lambda text: GetProductOutput.model_validate(json.loads(text)),
        )

    async def list_categories(
        self,
        client_ip: str,
        *,
        depth: int = 1,
    ) -> ListCategoriesOutput:
        """List category tree with rate limit and read cache."""
        categories_input = ListCategoriesInput(depth=depth, response_format="json")
        cache_args = _cache_args(categories_input)

        return await self._cached_read(
            client_ip=client_ip,
            tool_name=LIST_CATEGORIES_TOOL,
            cache_args=cache_args,
            fetch=lambda: list_categories(self._mcp, depth=depth),
            to_cache=lambda result: result.model_dump_json(),
            from_cache=lambda text: ListCategoriesOutput.model_validate(json.loads(text)),
        )

    async def list_delivery_cities(
        self,
        client_ip: str,
        *,
        query: str | None = None,
        limit: int = 25,
    ) -> list[str]:
        """List deliverable cities with rate limit and read cache."""
        cities_input = ListDeliveryCitiesInput(
            query=query,
            limit=limit,
            response_format="json",
        )
        cache_args = _cache_args(cities_input)

        def _to_cache(names: list[str]) -> str:
            output = ListDeliveryCitiesOutput(
                cities=[DeliveryCity(name=name) for name in names],
                total_matched=len(names),
                showing=len(names),
            )
            return output.model_dump_json()

        def _from_cache(text: str) -> list[str]:
            output = ListDeliveryCitiesOutput.model_validate(json.loads(text))
            return [city.name for city in output.cities]

        return await self._cached_read(
            client_ip=client_ip,
            tool_name=LIST_CITIES_TOOL,
            cache_args=cache_args,
            fetch=lambda: list_delivery_cities(self._mcp, query=query, limit=limit),
            to_cache=_to_cache,
            from_cache=_from_cache,
        )

    async def check_delivery(
        self,
        client_ip: str,
        *,
        city: str,
        delivery_date: str | None = None,
        product_id: str | None = None,
    ) -> CheckDeliveryOutput:
        """Check delivery availability with rate limit and read cache."""
        delivery_input = CheckDeliveryInput(
            city=city,
            delivery_date=delivery_date,
            product_id=product_id,
            response_format="json",
        )
        cache_args = _cache_args(delivery_input)

        return await self._cached_read(
            client_ip=client_ip,
            tool_name=CHECK_DELIVERY_TOOL,
            cache_args=cache_args,
            fetch=lambda: check_delivery(
                self._mcp,
                city=city,
                delivery_date=delivery_date,
                product_id=product_id,
            ),
            to_cache=lambda result: result.model_dump_json(),
            from_cache=lambda text: CheckDeliveryOutput.model_validate(json.loads(text)),
        )

    async def track_order(
        self,
        client_ip: str,
        *,
        order_number: str,
    ) -> TrackOrderOutput:
        """Track order status with rate limit and read cache."""
        track_input = TrackOrderInput(order_number=order_number, response_format="json")
        cache_args = _cache_args(track_input)

        return await self._cached_read(
            client_ip=client_ip,
            tool_name=TRACK_ORDER_TOOL,
            cache_args=cache_args,
            fetch=lambda: track_order(self._mcp, order_number=order_number),
            to_cache=lambda result: result.model_dump_json(),
            from_cache=lambda text: TrackOrderOutput.model_validate(json.loads(text)),
        )

    async def create_order(
        self,
        client_ip: str,
        *,
        cart: list[CartItem],
        recipient: Recipient,
        delivery: Delivery,
        sender: Sender,
        gift_message: str | None = None,
        currency: str = "LKR",
    ) -> CreateOrderResponse:
        """Create checkout order; rate-limited and never read-cached."""
        await check_rate_limit(self._redis, client_ip, CREATE_ORDER_TOOL)

        return await create_order(
            self._mcp,
            cart=cart,
            recipient=recipient,
            delivery=delivery,
            sender=sender,
            gift_message=gift_message,
            currency=currency,
        )
