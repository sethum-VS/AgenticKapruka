"""Synthesize assistant reply from MCP tool results and render HTMX partial."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.templating import get_templates
from graphs.nodes.analyze_intent import _extract_latest_user_message, create_genai_client
from graphs.state import AgentState, ModelTier

logger = logging.getLogger(__name__)

FLASH_MODEL = "gemini-2.5-flash"
PRO_MODEL = "gemini-2.5-pro"

SYSTEM_INSTRUCTION = """You are the Kapruka gift shopping assistant.

Synthesize a helpful, concise reply for the customer using ONLY the tool_results JSON provided.

Rules:
- Never invent products, prices, stock status, categories, or delivery facts.
- Quote product names and prices exactly as they appear in tool_results.
- If tool_results are empty or contain no useful data, say so politely and suggest next steps.
- Keep the reply conversational and under 200 words unless listing several products.
"""


class AssistantReply(BaseModel):
    """Structured Gemini response for the assistant message body."""

    message: str


def _select_model(state: AgentState) -> str:
    """Pick Gemini model from state tier; full router lands in PRD-032."""
    tier: ModelTier | None = state.get("model_tier")
    if tier == "pro":
        return PRO_MODEL
    return FLASH_MODEL


def _format_tool_results_context(tool_results: dict[str, Any] | None) -> str:
    """Serialize tool_results for the LLM context block."""
    payload = tool_results or {}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _build_user_prompt(user_message: str, tool_results: dict[str, Any] | None) -> str:
    """Combine user turn and MCP payload for response synthesis."""
    context = _format_tool_results_context(tool_results)
    return (
        f"Customer message:\n{user_message}\n\n"
        f"tool_results (sole source of truth for catalog facts):\n{context}"
    )


def _parse_reply_response(response: types.GenerateContentResponse) -> str:
    """Parse structured or JSON text assistant reply from Gemini."""
    if response.parsed is not None:
        if isinstance(response.parsed, AssistantReply):
            return response.parsed.message.strip()
        validated = AssistantReply.model_validate(response.parsed)
        return validated.message.strip()

    raw_text = (response.text or "").strip()
    if not raw_text:
        msg = "Gemini returned empty assistant reply"
        raise ValueError(msg)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        msg = f"Gemini reply is not valid JSON: {raw_text!r}"
        raise ValueError(msg) from exc

    try:
        return AssistantReply.model_validate(payload).message.strip()
    except ValidationError as exc:
        msg = f"Gemini reply JSON failed validation: {payload!r}"
        raise ValueError(msg) from exc


def _generate_reply_sync(
    client: genai.Client,
    *,
    model: str,
    user_prompt: str,
) -> str:
    """Blocking Gemini call; run via asyncio.to_thread from generate_response."""
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=AssistantReply,
            temperature=0.2,
        ),
    )
    return _parse_reply_response(response)


def render_assistant_html(message: str) -> str:
    """Render templates/chat/message_assistant.html for HTMX swap."""
    templates = get_templates()
    template = templates.env.get_template("chat/message_assistant.html")
    return template.render(message=message)


async def generate_response(
    state: AgentState,
    *,
    genai_client: genai.Client | None = None,
) -> dict[str, Any]:
    """LangGraph node: synthesize assistant text and render response_html partial."""
    messages = state.get("messages") or []
    user_message = _extract_latest_user_message(messages)
    tool_results = state.get("tool_results")

    if not user_message.strip():
        fallback = "How can I help you find a gift on Kapruka today?"
        return {"response_html": render_assistant_html(fallback)}

    client = genai_client or create_genai_client()
    model = _select_model(state)
    user_prompt = _build_user_prompt(user_message, tool_results)

    reply_text = await asyncio.to_thread(
        _generate_reply_sync,
        client,
        model=model,
        user_prompt=user_prompt,
    )

    if not reply_text:
        reply_text = "I could not generate a response. Please try again."

    logger.info("generate_response: rendered assistant reply (%d chars)", len(reply_text))
    return {"response_html": render_assistant_html(reply_text)}
