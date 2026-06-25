"""LLM-driven shipment destination extraction with Kapruka city cross-check."""

from __future__ import annotations

import logging
import re
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from graphs.model_router import FLASH_MODEL
from graphs.nodes.analyze_intent import _extract_latest_user_message
from graphs.state import AgentState
from lib.chat.city_resolution import resolve_delivery_city
from lib.chat.query_preprocessor import extract_target_city
from lib.genai.fallback import generate_content_with_fallback
from lib.kapruka.service import KaprukaService
from lib.zep.memory import format_memory_facts_block

logger = logging.getLogger(__name__)

_CONFIRM_RE = re.compile(
    r"\b(?:yes|yeah|yep|correct|that'?s right|confirm|ok(?:ay)?)\b",
    re.I,
)

_ADDRESS_SYSTEM = """Extract the Kapruka delivery destination from the customer message.

Return structured JSON with:
- raw_text: verbatim destination phrase from the message (or empty)
- city_candidate: best Kapruka delivery city or zone (e.g. Colombo 03, Galle)
- zone: Colombo zone number when present (e.g. 03), else null
- country: country when mentioned, default Sri Lanka
- confidence: high | medium | low

Support English, Sinhala script, and Tanglish. Prefer specific Colombo zones over bare Colombo.
"""


class ExtractedDestination(BaseModel):
    """Structured Gemini extraction for a shipment destination."""

    raw_text: str = ""
    city_candidate: str | None = None
    zone: str | None = None
    country: str = "Sri Lanka"
    confidence: str = Field(default="medium")


def _user_confirms_destination(user_message: str) -> bool:
    return bool(_CONFIRM_RE.search(user_message.strip()))


def _user_picks_candidate(user_message: str, candidates: list[str]) -> str | None:
    lowered = user_message.strip().lower()
    for candidate in candidates:
        if candidate.strip().lower() in lowered or lowered in candidate.strip().lower():
            return candidate
    return None


def extract_destination_regex(user_message: str) -> str | None:
    """Regex fallback for city extraction."""
    return extract_target_city(user_message)


async def extract_destination_llm(
    client: genai.Client | None,
    user_message: str,
    *,
    memory_facts: list[str] | None = None,
) -> ExtractedDestination | None:
    """Gemini Flash structured extraction for free-text shipment addresses."""
    if client is None or not user_message.strip():
        return None

    instruction = _ADDRESS_SYSTEM
    if memory_facts:
        instruction += format_memory_facts_block(memory_facts)

    try:
        response = generate_content_with_fallback(
            client=client,
            model=FLASH_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                response_mime_type="application/json",
                response_schema=ExtractedDestination,
                temperature=0,
            ),
        )
    except Exception:
        logger.warning("extract_destination_llm: Gemini call failed", exc_info=True)
        return None

    if response.parsed is not None:
        try:
            if isinstance(response.parsed, ExtractedDestination):
                return response.parsed
            return ExtractedDestination.model_validate(response.parsed)
        except ValidationError:
            return None

    raw_text = (response.text or "").strip()
    if not raw_text:
        return None
    try:
        return ExtractedDestination.model_validate_json(raw_text)
    except Exception:
        logger.debug("extract_destination_llm: invalid JSON %r", raw_text)
        return None


async def resolve_shipment_address(
    state: AgentState,
    *,
    kapruka_service: KaprukaService,
    client_ip: str,
    genai_client: genai.Client | None = None,
) -> dict[str, Any]:
    """Resolve shipment destination via LLM + MCP; ask user to confirm when ambiguous."""
    user_message = _extract_latest_user_message(state.get("messages") or [])
    candidates = list(state.get("delivery_city_candidates") or [])
    prior_raw = state.get("delivery_city_raw")
    session_confirmed = bool(state.get("session_delivery_city_confirmed"))

    if session_confirmed and state.get("session_delivery_city_canonical"):
        return {}

    if candidates and (_user_confirms_destination(user_message) or prior_raw):
        picked = _user_picks_candidate(user_message, candidates)
        if picked or _user_confirms_destination(user_message):
            canonical = picked or state.get("delivery_city_canonical")
            if isinstance(canonical, str) and canonical.strip():
                return {
                    "delivery_city_canonical": canonical.strip(),
                    "session_delivery_city_canonical": canonical.strip(),
                    "delivery_city_status": "resolved",
                    "session_delivery_city_confirmed": True,
                    "delivery_context_ready": True,
                }

    extracted = await extract_destination_llm(
        genai_client,
        user_message,
        memory_facts=state.get("zep_memory_facts"),
    )
    city_candidate = None
    raw_text = None
    confidence = "medium"

    if extracted is not None:
        city_candidate = (extracted.city_candidate or "").strip() or None
        raw_text = (extracted.raw_text or city_candidate or "").strip() or None
        confidence = (extracted.confidence or "medium").lower()
    if not city_candidate:
        city_candidate = extract_destination_regex(user_message)
        raw_text = raw_text or city_candidate

    if not city_candidate:
        return {}

    resolution = await resolve_delivery_city(kapruka_service, client_ip, city_candidate)

    updates: dict[str, Any] = {
        "delivery_city_raw": raw_text or city_candidate,
        "delivery_city_status": resolution.status,
        "delivery_city_candidates": resolution.candidates,
    }
    if raw_text:
        updates["session_shipment_address_raw"] = raw_text[:250]

    if resolution.status == "resolved" and resolution.canonical:
        regex_hit = bool(extract_destination_regex(user_message))
        if confidence == "high" or (regex_hit and extracted is None):
            updates.update(
                {
                    "delivery_city_canonical": resolution.canonical,
                    "session_delivery_city_canonical": resolution.canonical,
                    "session_delivery_city_confirmed": True,
                    "delivery_context_ready": True,
                },
            )
            return updates

        numbered = resolution.candidates or [resolution.canonical]
        options = "\n".join(
            f"{idx}. {name}" for idx, name in enumerate(numbered[:5], start=1)
        )
        updates.update(
            {
                "delivery_context_ready": False,
                "agent_clarifying_question": (
                    f"I found delivery to **{resolution.canonical}**. "
                    f"Is that correct?\n\n{options}\n\n"
                    "Reply with the number or city name to confirm."
                ),
                "agent_loop_exit_reason": "ask_user",
            },
        )
        return updates

    if resolution.status == "ambiguous":
        updates.update(
            {
                "delivery_context_ready": False,
                "agent_clarifying_question": resolution.customer_message
                or "Which delivery area did you mean?",
                "agent_loop_exit_reason": "ask_user",
            },
        )
        return updates

    if resolution.status == "not_found":
        updates.update(
            {
                "delivery_context_ready": False,
                "agent_clarifying_question": resolution.customer_message
                or "Which Kapruka delivery city should we use?",
                "agent_loop_exit_reason": "ask_user",
            },
        )
        return updates

    return updates
