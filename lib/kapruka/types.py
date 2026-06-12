"""Pydantic v2 models for Kapruka MCP tool inputs and outputs."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from lib.utils.text import decode_html_entities

SUPPORTED_CURRENCIES = frozenset({"LKR", "USD", "GBP", "AUD", "CAD", "EUR"})
LOCATION_TYPES = frozenset({"house", "apartment", "office", "other"})
SEARCH_SORT_VALUES = frozenset({"relevance", "price_asc", "price_desc", "newest", "bestseller"})
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SL_MOBILE_PHONE = re.compile(r"^(?:\+9477\d{7}|077\d{7})$")

ResponseFormat = Literal["markdown", "json"]


class Money(BaseModel):
    """Monetary amount with currency code."""

    amount: float | None = None
    currency: str


class CategoryRef(BaseModel):
    """Product category reference in search/detail payloads."""

    id: str
    name: str
    slug: str
    path: str | None = None


# --- kapruka_search_products ---


class SearchProductsInput(BaseModel):
    """Input for kapruka_search_products."""

    q: str = Field(..., min_length=3, max_length=200)
    category: str | None = None
    limit: int = Field(default=10, ge=1, le=50)
    cursor: str | None = None
    currency: str = "LKR"
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    in_stock_only: bool = False
    sort: str = "relevance"
    include_stubs: bool = False
    response_format: ResponseFormat = "json"

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        upper = value.upper()
        if upper not in SUPPORTED_CURRENCIES:
            msg = f"currency must be one of: {', '.join(sorted(SUPPORTED_CURRENCIES))}"
            raise ValueError(msg)
        return upper

    @field_validator("sort")
    @classmethod
    def validate_sort(cls, value: str) -> str:
        if value not in SEARCH_SORT_VALUES:
            msg = f"sort must be one of: {', '.join(sorted(SEARCH_SORT_VALUES))}"
            raise ValueError(msg)
        return value


class ProductResult(BaseModel):
    """Single product in search results."""

    id: str
    name: str
    summary: str
    price: Money
    compare_at_price: Money | None = None
    in_stock: bool
    stock_level: str
    image_url: str | None = None
    category: CategoryRef
    rating: float | None = None
    ships_internationally: bool
    url: str

    @field_validator("name", "summary")
    @classmethod
    def decode_html_entities_in_text(cls, value: str) -> str:
        return decode_html_entities(value)


class SearchProductsOutput(BaseModel):
    """Output from kapruka_search_products."""

    results: list[ProductResult]
    next_cursor: str | None = None
    applied_filters: dict[str, object]


# --- kapruka_get_product ---


class GetProductInput(BaseModel):
    """Input for kapruka_get_product."""

    product_id: str = Field(..., min_length=3, max_length=80)
    currency: str = "LKR"
    type: str | None = None
    response_format: ResponseFormat = "json"

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        upper = value.upper()
        if upper not in SUPPORTED_CURRENCIES:
            msg = f"currency must be one of: {', '.join(sorted(SUPPORTED_CURRENCIES))}"
            raise ValueError(msg)
        return upper


class ProductVariant(BaseModel):
    """Purchasable variant on a product detail page."""

    id: str
    name: str
    sku: str
    price: Money
    in_stock: bool
    stock_level: str
    attributes: dict[str, str] = Field(default_factory=dict)


class ProductAttributes(BaseModel):
    """Structured product attributes from Kapruka catalog."""

    type: str | None = None
    subtype: str | None = None
    weight: str | None = None
    vendor: str | None = None


class ProductShipping(BaseModel):
    """Shipping constraints for a product."""

    ships_from: str
    ships_internationally: bool
    restricted_countries: list[str] = Field(default_factory=list)


class GetProductOutput(BaseModel):
    """Output from kapruka_get_product."""

    id: str
    name: str
    description: str
    summary: str
    price: Money
    compare_at_price: Money | None = None
    in_stock: bool
    stock_level: str
    category: CategoryRef
    variants: list[ProductVariant] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    attributes: ProductAttributes
    shipping: ProductShipping
    rating: float | None = None
    url: str
    description_format: str | None = None

    @field_validator("name", "description", "summary")
    @classmethod
    def decode_html_entities_in_text(cls, value: str) -> str:
        return decode_html_entities(value)


# --- kapruka_list_categories ---


class ListCategoriesInput(BaseModel):
    """Input for kapruka_list_categories."""

    depth: int = Field(default=1, ge=1, le=2)
    response_format: ResponseFormat = "json"


class CategoryNode(BaseModel):
    """Category tree node; children populated when depth > 1."""

    name: str
    url: str
    children: list[CategoryNode] = Field(default_factory=list)


class ListCategoriesOutput(BaseModel):
    """Output from kapruka_list_categories."""

    categories: list[CategoryNode]


# --- kapruka_list_delivery_cities ---


class ListDeliveryCitiesInput(BaseModel):
    """Input for kapruka_list_delivery_cities."""

    query: str | None = Field(default=None, max_length=50)
    limit: int = Field(default=25, ge=1, le=50)
    response_format: ResponseFormat = "json"


class DeliveryCity(BaseModel):
    """Canonical deliverable city with optional aliases."""

    name: str
    aliases: list[str] = Field(default_factory=list)


class ListDeliveryCitiesOutput(BaseModel):
    """Output from kapruka_list_delivery_cities."""

    cities: list[DeliveryCity]
    total_matched: int
    showing: int


# --- kapruka_check_delivery ---


class CheckDeliveryInput(BaseModel):
    """Input for kapruka_check_delivery."""

    city: str = Field(..., min_length=2, max_length=100)
    delivery_date: str | None = None
    product_id: str | None = Field(default=None, max_length=80)
    response_format: ResponseFormat = "json"

    @field_validator("delivery_date")
    @classmethod
    def validate_delivery_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ISO_DATE.match(value):
            msg = "delivery_date must be YYYY-MM-DD"
            raise ValueError(msg)
        date.fromisoformat(value)
        return value


class CheckDeliveryOutput(BaseModel):
    """Output from kapruka_check_delivery."""

    city: str
    now: str
    checked_date: str
    available: bool
    rate: float
    currency: str
    reason: str | None = None
    next_available_date: str | None = None
    perishable_warning: str | None = None


# --- kapruka_create_order ---


class CartItem(BaseModel):
    """Line item in a Kapruka checkout cart."""

    product_id: str = Field(..., min_length=3, max_length=80)
    quantity: int = Field(default=1, ge=1, le=99)
    icing_text: str | None = Field(default=None, max_length=120)


class Recipient(BaseModel):
    """Gift recipient contact details."""

    name: str = Field(..., min_length=1, max_length=80)
    phone: str = Field(..., min_length=7, max_length=30)

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: str) -> str:
        phone = value.strip()
        if not _SL_MOBILE_PHONE.match(phone):
            msg = "Phone must be E.164 +9477XXXXXXX or local 077XXXXXXX format"
            raise ValueError(msg)
        return phone


class Delivery(BaseModel):
    """Delivery destination and schedule."""

    address: str = Field(..., min_length=3, max_length=250)
    city: str = Field(..., min_length=2, max_length=100)
    location_type: str = "house"
    date: str = Field(..., min_length=10, max_length=10)
    instructions: str | None = Field(default=None, max_length=250)

    @field_validator("location_type")
    @classmethod
    def validate_location_type(cls, value: str) -> str:
        lower = value.lower()
        if lower not in LOCATION_TYPES:
            msg = f"location_type must be one of: {', '.join(sorted(LOCATION_TYPES))}"
            raise ValueError(msg)
        return lower

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        if not _ISO_DATE.match(value):
            msg = "date must be YYYY-MM-DD"
            raise ValueError(msg)
        date.fromisoformat(value)
        return value


class Sender(BaseModel):
    """Sender name shown on the gift card."""

    name: str = Field(..., min_length=1, max_length=80)
    anonymous: bool = False


class CreateOrderInput(BaseModel):
    """Input for kapruka_create_order."""

    cart: list[CartItem] = Field(..., min_length=1, max_length=30)
    recipient: Recipient
    delivery: Delivery
    sender: Sender
    gift_message: str | None = Field(default=None, max_length=300)
    currency: str = "LKR"
    response_format: ResponseFormat = "json"

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        upper = value.upper()
        if upper not in SUPPORTED_CURRENCIES:
            msg = f"currency must be one of: {', '.join(sorted(SUPPORTED_CURRENCIES))}"
            raise ValueError(msg)
        return upper


class OrderSummary(BaseModel):
    """Price breakdown returned with a checkout link."""

    items_total: float
    delivery_fee: float
    addons_total: float
    grand_total: float
    currency: str


class CreateOrderResponse(BaseModel):
    """Output from kapruka_create_order."""

    checkout_url: str
    order_ref: str
    summary: OrderSummary
    expires_at: str


# --- kapruka_track_order ---


class TrackOrderInput(BaseModel):
    """Input for kapruka_track_order."""

    order_number: str = Field(..., min_length=4, max_length=40)
    response_format: ResponseFormat = "json"


class TrackOrderRecipient(BaseModel):
    """Recipient details on a tracked order."""

    name: str
    phone: str
    address: str
    city: str


class TrackOrderProgressEvent(BaseModel):
    """Single step in an order progress timeline."""

    step: str
    timestamp: str


class TrackOrderItem(BaseModel):
    """Line item on a tracked order."""

    product_id: str
    name: str
    quantity: int
    selling_price: float


def _format_track_amount_currency_code(raw_amount: float | int | str, currency: str) -> str:
    """Format MCP money as a display string (e.g. ``LKR 4,970``)."""
    code = currency.upper()
    num = float(str(raw_amount).replace(",", ""))
    formatted = f"{round(num):,}" if code == "LKR" else f"{num:,.2f}"
    return f"{code} {formatted}"


def coerce_track_order_amount(value: Any) -> str:
    """Normalize track-order amount from MCP string or Money-shaped payload."""
    if isinstance(value, str):
        return value

    if isinstance(value, Money):
        raw_amount = value.amount
        currency = value.currency
    elif isinstance(value, dict):
        currency = str(value.get("currency") or "LKR")
        raw_amount = value.get("value", value.get("amount"))
    else:
        msg = f"amount must be a string or money object, got {type(value).__name__}"
        raise ValueError(msg)

    if raw_amount is None:
        msg = "amount money object missing value/amount"
        raise ValueError(msg)

    return _format_track_amount_currency_code(raw_amount, currency)


class TrackOrderOutput(BaseModel):
    """Output from kapruka_track_order."""

    order_number: str
    pnref: str
    status: str
    status_display: str
    order_date: str
    delivery_date: str
    shipped_date: str | None = None
    amount: str
    payment_method: str
    comments: str | None = None
    recipient: TrackOrderRecipient
    greeting_message: str | None = None
    special_instructions: str | None = None
    progress: list[TrackOrderProgressEvent] = Field(default_factory=list)
    live_tracking_available: bool
    has_delivery_video: bool
    has_delivery_photo: bool
    items: list[TrackOrderItem] = Field(default_factory=list)

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, value: Any) -> str:
        return coerce_track_order_amount(value)


CategoryNode.model_rebuild()
