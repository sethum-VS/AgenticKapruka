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


ACTIVE_PATCHERS = []

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

    async def fake_generate_content(
        *,
        model: str | None = None,
        messages: list[Any],
        response_schema: Any = None,
        **kwargs: Any,
    ) -> Any:
        nonlocal planner_calls
        schema_name = getattr(response_schema, "__name__", "")

        if schema_name == "IntentClassification":
            resolved_intent = _resolve_intent()
            from graphs.nodes.analyze_intent import IntentClassification
            return IntentClassification(intent=resolved_intent)

        if schema_name == "AgentPlannerStep":
            planner_calls += 1
            resolved_intent = _resolve_intent()
            from graphs.nodes.agent_loop import AgentPlannerStep
            if planner_calls == 1 and resolved_intent == "general":
                return AgentPlannerStep(
                    action="finish",
                    refined_intent="general",
                    rationale="no catalog tools needed",
                )
            elif planner_calls == 1:
                query = search_query or "gifts"
                if messages and messages[-1].get("content"):
                    q = messages[-1]["content"].strip()
                    if len(q) >= 3:
                        query = q
                from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
                return AgentPlannerStep(
                    action="call_tool",
                    tool_name=SEARCH_PRODUCTS_TOOL,
                    tool_args={"q": query},
                    refined_intent="discovery",
                    rationale="search catalog",
                )
            else:
                return AgentPlannerStep(action="finish", rationale="catalog facts collected")

        if schema_name == "AssistantReply":
            from graphs.nodes.generate_response import AssistantReply
            return AssistantReply(message=assistant_message)

        if schema_name == "MasterFlowAlignment":
            from lib.chat.master_flow import MasterFlowAlignment
            return MasterFlowAlignment(
                decision="hold",
                confidence=0.9,
                active_flow="shopping",
            )

        return {"content": "mocked", "role": "assistant"}

    from lib.genai.completions import set_override_generate_content
    set_override_generate_content(fake_generate_content)
    ACTIVE_PATCHERS.append(fake_generate_content)
    return mock_client

