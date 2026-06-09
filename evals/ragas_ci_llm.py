"""Deterministic LangChain chat model for Ragas metrics in CI (no live judge LLM)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class CiRagasChatModel(BaseChatModel):
    """Return Ragas-compatible JSON so CI can score metrics without API keys."""

    @property
    def _llm_type(self) -> str:
        return "ci-ragas"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = stop, run_manager, kwargs
        return self._build_chat_result(messages)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Native async path for Ragas metric jobs (avoids executor deadlocks in CI)."""
        _ = stop, run_manager, kwargs
        return self._build_chat_result(messages)

    def _build_chat_result(self, messages: list[BaseMessage]) -> ChatResult:
        prompt = str(messages[-1].content)
        content = self._response_for_prompt(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _response_for_prompt(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "break down each sentence" in lowered:
            return json.dumps(
                {
                    "statements": [
                        "The assistant summarized Kapruka catalog data from the tool results.",
                    ],
                },
            )
        if "judge the faithfulness" in lowered:
            return json.dumps(
                {
                    "statements": [
                        {
                            "statement": "Catalog summary statement",
                            "reason": "Supported by retrieved MCP context.",
                            "verdict": 1,
                        },
                    ],
                },
            )
        if "noncommittal" in lowered:
            return json.dumps({"question": "What is the customer looking for?", "noncommittal": 0})
        if "verify if the context was useful" in lowered or (
            "verdict" in lowered and "context" in lowered
        ):
            return json.dumps(
                {
                    "reason": "Retrieved MCP context supports the assistant answer.",
                    "verdict": 1,
                },
            )
        if "fix the output string" in lowered:
            return json.dumps({"text": '{"verdict": 1, "reason": "ok"}'})
        return json.dumps({"verdict": 1, "reason": "ok"})
