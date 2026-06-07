"""Sender form parsing and Pydantic validation."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from lib.kapruka.types import Sender


@dataclass(frozen=True)
class SenderFormValues:
    """Raw sender form field values preserved across HTMX validation swaps."""

    name: str = ""
    anonymous: bool = False


def parse_sender_form(values: SenderFormValues) -> tuple[Sender | None, dict[str, str]]:
    """Validate sender form fields via the Kapruka Sender model.

    Returns (parsed_sender, field_errors). field_errors maps form field names to messages.
    """
    payload = {
        "name": values.name.strip(),
        "anonymous": values.anonymous,
    }
    try:
        sender = Sender(**payload)
    except ValidationError as exc:
        errors: dict[str, str] = {}
        for err in exc.errors():
            field = str(err["loc"][0])
            errors[field] = str(err["msg"])
        return None, errors
    return sender, {}
