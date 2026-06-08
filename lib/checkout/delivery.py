"""Delivery address form parsing and Pydantic validation."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from lib.kapruka.types import Delivery


@dataclass(frozen=True)
class DeliveryFormValues:
    """Raw delivery form field values preserved across HTMX validation swaps."""

    address: str = ""
    city: str = ""
    location_type: str = "house"
    date: str = ""
    instructions: str = ""


def parse_delivery_form(values: DeliveryFormValues) -> tuple[Delivery | None, dict[str, str]]:
    """Validate delivery form fields via the Kapruka Delivery model.

    Returns (parsed_delivery, field_errors). field_errors maps form field names to messages.
    """
    instructions = values.instructions.strip() or None
    payload = {
        "address": values.address.strip(),
        "city": values.city.strip(),
        "location_type": values.location_type.strip() or "house",
        "date": values.date.strip(),
        "instructions": instructions,
    }
    try:
        delivery = Delivery(**payload)
    except ValidationError as exc:
        errors: dict[str, str] = {}
        for err in exc.errors():
            field = str(err["loc"][0])
            errors[field] = str(err["msg"])
        return None, errors
    return delivery, {}
