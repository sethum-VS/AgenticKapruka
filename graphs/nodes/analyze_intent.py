"""Classify user intent via Gemini structured output."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from graphs.state import AgentState, Intent
from lib.zep.memory import format_memory_facts_block

logger = logging.getLogger(__name__)

INTENT_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """You classify user messages for the Kapruka gift shopping assistant.

Return exactly one intent:
- discovery: browsing, searching, or finding gifts and products
- checkout: placing an order, cart, delivery, payment, or recipient details
- tracking: order status, delivery progress, or locating an existing order
- general: greetings, thanks, FAQ, or unclear/off-topic messages
"""

VALID_INTENTS: frozenset[Intent] = frozenset(
    {"discovery", "checkout", "tracking", "general"},
)


class IntentClassification(BaseModel):
    """Structured Gemini response for intent routing."""

    intent: Intent


def create_genai_client(*, api_key: str | None = None) -> genai.Client:
    """Build a google-genai client; inject api_key in tests."""
    key = api_key if api_key is not None else get_settings().google_api_key
    return genai.Client(api_key=key)


def _extract_latest_user_message(messages: list[BaseMessage]) -> str:
    """Return content of the most recent human message."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                return " ".join(parts).strip()
            return str(content)
    return ""


def _parse_intent_response(response: types.GenerateContentResponse) -> Intent:
    """Parse structured or JSON text intent from a Gemini response."""
    if response.parsed is not None:
        if isinstance(response.parsed, IntentClassification):
            return response.parsed.intent
        validated = IntentClassification.model_validate(response.parsed)
        return validated.intent

    raw_text = (response.text or "").strip()
    if not raw_text:
        msg = "Gemini returned empty intent classification"
        raise ValueError(msg)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        msg = f"Gemini intent response is not valid JSON: {raw_text!r}"
        raise ValueError(msg) from exc

    try:
        return IntentClassification.model_validate(payload).intent
    except ValidationError as exc:
        msg = f"Gemini intent JSON failed validation: {payload!r}"
        raise ValueError(msg) from exc


def _build_intent_system_instruction(zep_memory_facts: list[str] | None) -> str:
    """Combine base intent prompt with optional Zep memory context."""
    instruction = SYSTEM_INSTRUCTION
    if zep_memory_facts:
        instruction += format_memory_facts_block(zep_memory_facts)
    return instruction


def _classify_intent_sync(
    client: genai.Client,
    user_message: str,
    *,
    zep_memory_facts: list[str] | None = None,
) -> Intent:
    """Blocking Gemini call; run via asyncio.to_thread from analyze_intent."""
    response = client.models.generate_content(
        model=INTENT_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=_build_intent_system_instruction(zep_memory_facts),
            response_mime_type="application/json",
            response_schema=IntentClassification,
            temperature=0,
        ),
    )
    intent = _parse_intent_response(response)
    if intent not in VALID_INTENTS:
        msg = f"Unsupported intent value from model: {intent!r}"
        raise ValueError(msg)
    return intent


async def analyze_intent(
    state: AgentState,
    *,
    genai_client: genai.Client | None = None,
) -> dict[str, Any]:
    """LangGraph node: classify the latest user message into routing intent."""
    messages = state.get("messages") or []
    user_message = _extract_latest_user_message(messages)

    if not user_message.strip():
        logger.debug("analyze_intent: empty user message, defaulting to general")
        return {"intent": "general"}

    client = genai_client or create_genai_client()
    zep_memory_facts = state.get("zep_memory_facts")
    intent = await asyncio.to_thread(
        _classify_intent_sync,
        client,
        user_message,
        zep_memory_facts=zep_memory_facts,
    )
    logger.info("analyze_intent: classified message as %s", intent)
    return {"intent": intent}
