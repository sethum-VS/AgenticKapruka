"""Shared Kapruka delivery city resolution for chat and checkout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from lib.kapruka.service import KaprukaService

CityResolutionStatus = Literal["resolved", "ambiguous", "not_found", "missing"]

_COLOMBO_ZONE = re.compile(r"^colombo\s+\d{2}$", re.I)
_AMBIGUOUS_CANDIDATE_LIMIT = 5


@dataclass(frozen=True, slots=True)
class CityResolution:
    """Outcome of resolving a raw city string against Kapruka delivery cities."""

    status: CityResolutionStatus
    canonical: str | None = None
    candidates: list[str] | None = None
    customer_message: str | None = None


def _is_bare_colombo(raw_city: str) -> bool:
    """True when the customer named Colombo without a zone number."""
    parts = raw_city.strip().split()
    return len(parts) == 1 and parts[0].lower() == "colombo"


def _exact_match(raw_city: str, cities: list[str]) -> str | None:
    lowered = raw_city.strip().lower()
    for city in cities:
        if city.strip().lower() == lowered:
            return city
    return None


def _ambiguous_colombo_message(candidates: list[str]) -> str:
    shown = candidates[:_AMBIGUOUS_CANDIDATE_LIMIT]
    if len(shown) == 1:
        examples = shown[0]
    elif len(shown) == 2:
        examples = f"{shown[0]} or {shown[1]}"
    else:
        head = ", ".join(shown[:-1])
        examples = f"{head}, or {shown[-1]}"
    return (
        "Colombo has several delivery zones. Which area should we deliver to? "
        f"For example: {examples}."
    )


def build_city_not_found_message() -> str:
    """Customer copy when Kapruka has no matching delivery city."""
    return (
        "I couldn't find that city in Kapruka's delivery network. "
        "Please try a nearby delivery area (for example Colombo 03, Kandy, or Galle)."
    )


async def resolve_delivery_city(
    service: KaprukaService,
    client_ip: str,
    raw_city: str | None,
    *,
    limit: int = 50,
) -> CityResolution:
    """Resolve a raw city string via kapruka_list_delivery_cities."""
    stripped = (raw_city or "").strip()
    if not stripped:
        return CityResolution(
            status="missing",
            customer_message="Which city should we deliver to?",
        )

    cities = await service.list_delivery_cities(client_ip, query=stripped, limit=limit)

    if _is_bare_colombo(stripped):
        colombo_zones = [city for city in cities if _COLOMBO_ZONE.match(city.strip())]
        if len(colombo_zones) > 1:
            candidates = colombo_zones[:_AMBIGUOUS_CANDIDATE_LIMIT]
            return CityResolution(
                status="ambiguous",
                candidates=candidates,
                customer_message=_ambiguous_colombo_message(candidates),
            )

    exact = _exact_match(stripped, cities)
    if exact is not None:
        return CityResolution(status="resolved", canonical=exact)

    if len(cities) == 1:
        return CityResolution(status="resolved", canonical=cities[0])

    if not cities:
        return CityResolution(status="not_found", customer_message=build_city_not_found_message())

    return CityResolution(status="not_found", customer_message=build_city_not_found_message())
