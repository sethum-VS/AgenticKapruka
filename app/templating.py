"""Jinja2 template environment for server-rendered HTMX pages."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import jinja2
from fastapi.templating import Jinja2Templates

from lib.checkout.delivery import DeliveryFormValues
from lib.checkout.payment import PaymentCtaContext
from lib.checkout.recipient import RecipientFormValues
from lib.checkout.review import CheckoutReviewContext
from lib.checkout.sender import SenderFormValues
from lib.kapruka.types import LOCATION_TYPES, CheckDeliveryOutput, TrackOrderOutput
from lib.redis.cart import StoredCartItem
from lib.utils.currency import SUPPORTED_CURRENCIES, format_currency
from lib.utils.text import decode_html_entities
from lib.utils.timezone import colombo_today_iso

SUPPORTED_CURRENCY_CODES: tuple[str, ...] = tuple(sorted(SUPPORTED_CURRENCIES))

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def urlencode_filter(value: str) -> str:
    """URL-encode query parameter values for HTMX hx-get URLs."""
    return quote(str(value), safe="")


def _is_dev_environment() -> bool:
    return os.getenv("APP_ENV", "development").lower() != "production"


@lru_cache
def _create_templates() -> Jinja2Templates:
    loader = jinja2.FileSystemLoader(str(TEMPLATES_DIR))
    env = jinja2.Environment(
        loader=loader,
        autoescape=jinja2.select_autoescape(),
        auto_reload=_is_dev_environment(),
    )
    env.filters["format_currency"] = format_currency
    env.filters["urlencode"] = urlencode_filter
    env.filters["decode_html"] = decode_html_entities
    return Jinja2Templates(env=env)


def get_templates() -> Jinja2Templates:
    """FastAPI dependency returning the shared Jinja2 template environment."""
    return _create_templates()


def render_stock_badge(*, in_stock: bool, stock_level: str = "high") -> str:
    """Render templates/components/stock_badge.html for product image overlays."""
    templates = get_templates()
    template = templates.env.get_template("components/stock_badge.html")
    return template.render(in_stock=in_stock, stock_level=stock_level)


def render_product_card(product: dict[str, Any]) -> str:
    """Render templates/components/product_card.html for carousel and search results."""
    templates = get_templates()
    template = templates.env.get_template("components/product_card.html")
    return template.render(product=product)


def render_product_carousel(products: list[dict[str, Any]]) -> str:
    """Render templates/components/product_carousel.html with product_card partials."""
    templates = get_templates()
    template = templates.env.get_template("components/product_carousel.html")
    return template.render(products=products)


def categories_for_chips(hybrid_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract deduplicated category rows from hybrid_context for chip rendering."""
    if not hybrid_context:
        return []

    seen: set[str] = set()
    chips: list[dict[str, Any]] = []
    for source in ("vector_hits", "categories"):
        for item in hybrid_context.get(source) or []:
            name = item.get("display_name")
            if not name or name in seen:
                continue
            seen.add(str(name))
            chips.append(dict(item))
    return chips


def render_category_chips(
    categories: list[dict[str, Any]],
    *,
    active_category: str | None = None,
) -> str:
    """Render templates/components/category_chips.html for hybrid_context category filters."""
    templates = get_templates()
    template = templates.env.get_template("components/category_chips.html")
    return template.render(categories=categories, active_category=active_category)


def render_cart_partial(
    *,
    items: list[StoredCartItem],
    currency: str = "LKR",
) -> str:
    """Render templates/checkout/cart_partial.html for HTMX outerHTML swaps."""
    templates = get_templates()
    template = templates.env.get_template("checkout/cart_partial.html")
    return template.render(items=items, currency=currency)


def render_cart_partial_oob(
    *,
    items: list[StoredCartItem],
    currency: str = "LKR",
) -> str:
    """Cart partial with hx-swap-oob for chat-driven add-to-cart confirmation."""
    partial = render_cart_partial(items=items, currency=currency)
    return partial.replace(
        'id="cart-panel"',
        'id="cart-panel" hx-swap-oob="outerHTML"',
        1,
    )


def render_cart_drawer(
    *,
    items: list[StoredCartItem],
    currency: str = "LKR",
) -> str:
    """Render templates/components/cart_drawer.html for header cart slide-over."""
    templates = get_templates()
    template = templates.env.get_template("components/cart_drawer.html")
    return template.render(
        cart_items=items,
        cart_item_count=sum(item.quantity for item in items),
        currency=currency,
    )


def render_delivery_city() -> str:
    """Render templates/checkout/delivery_city.html for checkout city autocomplete."""
    templates = get_templates()
    template = templates.env.get_template("checkout/delivery_city.html")
    return template.render()


def render_delivery_city_suggestions(cities: list[str]) -> str:
    """Render li suggestion items for HTMX swap into #delivery-city-suggestions."""
    templates = get_templates()
    template = templates.env.get_template("checkout/delivery_city_suggestions.html")
    return template.render(cities=cities)


def render_delivery_date(*, min_date: str | None = None) -> str:
    """Render templates/checkout/delivery_date.html with Colombo min date."""
    templates = get_templates()
    template = templates.env.get_template("checkout/delivery_date.html")
    return template.render(min_date=min_date or colombo_today_iso())


def render_delivery_date_status(*, result: CheckDeliveryOutput) -> str:
    """Render delivery availability partial after kapruka_check_delivery."""
    templates = get_templates()
    template = templates.env.get_template("checkout/delivery_date_status.html")
    return template.render(result=result)


def render_delivery_date_error(*, title: str, message: str) -> str:
    """Render user-friendly delivery date validation error partial."""
    templates = get_templates()
    template = templates.env.get_template("checkout/delivery_date_error.html")
    return template.render(title=title, message=message)


def render_delivery_field_error(*, field: str, message: str) -> str:
    """Render HTMX OOB inline field error for delivery form validation."""
    templates = get_templates()
    template = templates.env.get_template("checkout/delivery_field_error.html")
    return template.render(field=field, message=message)


def render_delivery_form(
    *,
    values: DeliveryFormValues | None = None,
    min_date: str | None = None,
    valid: bool = False,
) -> str:
    """Render templates/checkout/delivery_form.html with optional submitted values."""
    templates = get_templates()
    template = templates.env.get_template("checkout/delivery_form.html")
    return template.render(
        values=values or DeliveryFormValues(),
        min_date=min_date or colombo_today_iso(),
        location_types=sorted(LOCATION_TYPES),
        valid=valid,
    )


def render_delivery_form_validation_response(
    *,
    values: DeliveryFormValues,
    errors: dict[str, str],
    valid: bool = False,
    min_date: str | None = None,
) -> str:
    """Render delivery form plus OOB field error fragments (preserves form state)."""
    form_html = render_delivery_form(values=values, min_date=min_date, valid=valid)
    if not errors:
        return form_html
    oob_errors = "".join(
        render_delivery_field_error(field=field, message=message)
        for field, message in errors.items()
    )
    return form_html + oob_errors


def render_recipient_field_error(*, field: str, message: str) -> str:
    """Render HTMX OOB inline field error for recipient form validation."""
    templates = get_templates()
    template = templates.env.get_template("checkout/recipient_field_error.html")
    return template.render(field=field, message=message)


def render_recipient_form(
    *,
    values: RecipientFormValues | None = None,
    valid: bool = False,
) -> str:
    """Render templates/checkout/recipient_form.html with optional submitted values."""
    templates = get_templates()
    template = templates.env.get_template("checkout/recipient_form.html")
    return template.render(
        values=values or RecipientFormValues(),
        valid=valid,
    )


def render_recipient_form_validation_response(
    *,
    values: RecipientFormValues,
    errors: dict[str, str],
    valid: bool = False,
) -> str:
    """Render recipient form plus OOB field error fragments (preserves form state)."""
    form_html = render_recipient_form(values=values, valid=valid)
    if not errors:
        return form_html
    oob_errors = "".join(
        render_recipient_field_error(field=field, message=message)
        for field, message in errors.items()
    )
    return form_html + oob_errors


def render_sender_field_error(*, field: str, message: str) -> str:
    """Render HTMX OOB inline field error for sender form validation."""
    templates = get_templates()
    template = templates.env.get_template("checkout/sender_field_error.html")
    return template.render(field=field, message=message)


def render_sender_form(
    *,
    values: SenderFormValues | None = None,
    valid: bool = False,
) -> str:
    """Render templates/checkout/sender_form.html with optional submitted values."""
    templates = get_templates()
    template = templates.env.get_template("checkout/sender_form.html")
    return template.render(
        values=values or SenderFormValues(),
        valid=valid,
    )


def render_sender_form_validation_response(
    *,
    values: SenderFormValues,
    errors: dict[str, str],
    valid: bool = False,
) -> str:
    """Render sender form plus OOB field error fragments (preserves form state)."""
    form_html = render_sender_form(values=values, valid=valid)
    if not errors:
        return form_html
    oob_errors = "".join(
        render_sender_field_error(field=field, message=message) for field, message in errors.items()
    )
    return form_html + oob_errors


def render_checkout_review(*, review: CheckoutReviewContext) -> str:
    """Render templates/checkout/review.html order summary for the review step."""
    templates = get_templates()
    template = templates.env.get_template("checkout/review.html")
    return template.render(review=review)


def render_payment_cta(*, payment: PaymentCtaContext) -> str:
    """Render templates/checkout/payment_cta.html click-to-pay countdown CTA."""
    templates = get_templates()
    template = templates.env.get_template("checkout/payment_cta.html")
    return template.render(payment=payment)


def render_tracking_status(*, tracking: TrackOrderOutput) -> str:
    """Render templates/checkout/tracking_status.html order progress partial."""
    templates = get_templates()
    template = templates.env.get_template("checkout/tracking_status.html")
    return template.render(tracking=tracking)


def render_not_found_page() -> str:
    """Render templates/errors/not_found.html for unknown browser navigations."""
    templates = get_templates()
    template = templates.env.get_template("errors/not_found.html")
    return template.render(
        cart_items=[],
        cart_item_count=0,
        currency="LKR",
        supported_currencies=SUPPORTED_CURRENCY_CODES,
    )


def render_error_banner(
    *,
    error_code: str,
    message: str,
    title: str = "Unable to complete request",
) -> str:
    """Render templates/partials/error_banner.html for HTMX error swaps."""
    templates = get_templates()
    template = templates.env.get_template("partials/error_banner.html")
    return template.render(error_code=error_code, message=message, title=title)


def render_rate_limit_banner(
    *,
    error_code: str,
    message: str,
    retry_after_seconds: int,
    title: str = "Rate limit reached",
) -> str:
    """Render templates/partials/rate_limit_banner.html with Retry-After countdown."""
    templates = get_templates()
    template = templates.env.get_template("partials/rate_limit_banner.html")
    return template.render(
        error_code=error_code,
        message=message,
        title=title,
        retry_after_seconds=retry_after_seconds,
    )


def render_currency_selector(*, currency: str = "LKR") -> str:
    """Render templates/components/currency_selector.html for the site header."""
    templates = get_templates()
    template = templates.env.get_template("components/currency_selector.html")
    return template.render(
        currency=currency,
        supported_currencies=SUPPORTED_CURRENCY_CODES,
    )


def normalize_html_snapshot(html: str) -> str:
    """Collapse insignificant whitespace for stable snapshot comparisons."""
    collapsed = re.sub(r">\s+<", "><", html.strip())
    return re.sub(r"\s{2,}", " ", collapsed)
