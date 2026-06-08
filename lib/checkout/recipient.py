"""Recipient form parsing and Pydantic validation."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from lib.kapruka.types import Recipient


@dataclass(frozen=True)
class RecipientFormValues:
    """Raw recipient form field values preserved across HTMX validation swaps."""

    name: str = ""
    phone: str = ""


def parse_recipient_form(values: RecipientFormValues) -> tuple[Recipient | None, dict[str, str]]:
    """Validate recipient form fields via the Kapruka Recipient model.

    Returns (parsed_recipient, field_errors). field_errors maps form field names to messages.
    """
    payload = {
        "name": values.name.strip(),
        "phone": values.phone.strip(),
    }
    try:
        recipient = Recipient(**payload)
    except ValidationError as exc:
        errors: dict[str, str] = {}
        for err in exc.errors():
            field = str(err["loc"][0])
            errors[field] = str(err["msg"])
        return None, errors
    return recipient, {}
