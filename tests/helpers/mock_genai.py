"""Shared Gemini client mocks for graph integration tests."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from google.genai import types

from graphs.nodes.agent_loop import AgentPlannerStep
from graphs.nodes.analyze_intent import IntentClassification
from graphs.nodes.generate_response import AssistantReply
from graphs.state import Intent
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL


def build_mock_genai_client(
    *,
    intent: Intent | list[Intent] = "discovery",
    search_query: str | None = None,
    assistant_message: str = "Happy to help with your Kapruka gift search.",
) -> MagicMock:
    """Gemini mock routing intent classification, agent-loop planner, and synthesis."""
    mock_client = MagicMock()
    planner_calls = 0
    intent_calls = 0
    intent_sequence: list[Intent] = [intent] if isinstance(intent, str) else list(intent)

    def _resolve_intent() -> Intent:
        nonlocal intent_calls
        idx = min(intent_calls, len(intent_sequence) - 1)
        resolved = intent_sequence[idx]
        intent_calls += 1
        return resolved

    def generate_content(
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
        **kwargs: Any,
    ) -> MagicMock:
        nonlocal planner_calls
        _ = model, kwargs
        response = MagicMock()
        if config is not None and config.response_schema is IntentClassification:
            resolved_intent = _resolve_intent()
            response.parsed = IntentClassification(intent=resolved_intent)
            response.text = json.dumps({"intent": resolved_intent})
            return response

        if config is not None and config.response_schema is AgentPlannerStep:
            planner_calls += 1
            if planner_calls == 1:
                query = search_query or contents.strip()
                if len(query) < 3:
                    query = "gifts"
                step = AgentPlannerStep(
                    action="call_tool",
                    tool_name=SEARCH_PRODUCTS_TOOL,
                    tool_args={"q": query},
                    rationale="search catalog",
                )
            else:
                step = AgentPlannerStep(action="finish", rationale="catalog facts collected")
            response.parsed = step
            response.text = step.model_dump_json()
            return response

        if config is not None and config.response_schema is AssistantReply:
            response.parsed = AssistantReply(message=assistant_message)
            response.text = json.dumps({"message": assistant_message})
            return response

        resolved_intent = _resolve_intent()
        response.parsed = IntentClassification(intent=resolved_intent)
        response.text = json.dumps({"intent": resolved_intent})
        return response

    mock_client.models.generate_content.side_effect = generate_content
    return mock_client
