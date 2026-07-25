"""E2E application factory: fakeredis, mock MCP, and a hermetic NIM override.

The E2E server runs the *real* shopping/checkout graphs. Production nodes call
``lib.genai.completions.generate_content`` directly (not the injected
``genai_client``), so we install a deterministic override via
``set_override_generate_content`` instead of relying on a live NVIDIA NIM
endpoint. This keeps Playwright personas fast and offline — no 60s SSE hangs or
429s from the real model.
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import fakeredis.aioredis
from evals.ragas_eval import build_eval_genai_client
from fastapi import FastAPI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.key_registry import AsyncCheckpointKeyRegistry
from tests.fixtures.mcp_mock import MockMCPHttpClient
from tests.unit.test_settings import _VALID_ENV

from app.config import get_settings
from app.main import create_app
from app.routes import chat as chat_route
from graphs.shopping_graph import ShoppingGraphDeps
from lib.chat.deps import client_ip_from_request
from lib.chat.intent_heuristics import classify_routing_guard
from lib.chat.off_topic import is_impossible_catalog_request, is_off_topic_message
from lib.genai.completions import set_override_generate_content
from lib.kapruka.service import KaprukaService
from lib.kapruka.tools.delivery import CHECK_DELIVERY_TOOL
from lib.kapruka.tools.search_products import TOOL_NAME as SEARCH_PRODUCTS_TOOL
from lib.redis.client import RedisClient

E2E_PORT = 8080

# ── Hermetic NIM override ─────────────────────────────────────────────────────
_EMPATHY_TRIGGER = re.compile(
    r"\b(?:broke\s*up|break\s*up|heartbroken|heart\s*broken|grieving|"
    r"passed\s+away|funeral|so\s+sad|devastated)\b",
    re.I,
)
_SHOPPING_SIGNAL = re.compile(
    r"\b(?:cake|cakes|flower|flowers|rose|roses|bouquet|chocolate|chocolates|"
    r"hamper|hampers|voucher|combo|gift|gifts|anniversary|birthday|wedding|"
    r"present|under|below|budget|rs\.?|rupees?|lkr|\d{3,})\b",
    re.I,
)
_DELIVERY_SIGNAL = re.compile(
    r"\b(?:deliver|delivery|kandy|colombo|galle|sunday|saturday|monday|"
    r"tuesday|wednesday|thursday|friday|this\s+week|next\s+week)\b",
    re.I,
)
_CITY_SIGNAL = re.compile(
    r"\b(?P<city>Kandy|Colombo(?:\s*\d{2})?|Galle|Negombo|Jaffna)\b",
    re.I,
)


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the most recent user-role message content."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return str(messages[-1].get("content") or "") if messages else ""


def _combined_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content") or "") for message in messages)


def _planner_query(user_prompt: str) -> str:
    """Extract a search query from the planner's 'Customer message:' user prompt."""
    marker = "Customer message:"
    text = user_prompt
    if marker in text:
        text = text.split(marker, 1)[1]
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return first_line or "gifts"


def _customer_message_from_planner(user_prompt: str) -> str:
    return _planner_query(user_prompt)


def _resolve_intent(user_text: str) -> str:
    """Heuristic intent for LLM classification fallback (guards usually win first)."""
    guard = classify_routing_guard(user_text)
    if guard is not None:
        return guard
    if is_off_topic_message(user_text) or is_impossible_catalog_request(user_text):
        return "general"
    if _DELIVERY_SIGNAL.search(user_text) and not _SHOPPING_SIGNAL.search(user_text):
        return "discovery"
    return "discovery"


def _assistant_reply_copy(user_text: str) -> str:
    """Deterministic assistant body copy that satisfies persona tone assertions."""
    if _EMPATHY_TRIGGER.search(user_text):
        return (
            "I'm so sorry to hear that — that sounds really hard, and I'm here for you. "
            "Whenever you're ready, I can help you find a thoughtful gift, no rush at all."
        )
    if is_off_topic_message(user_text):
        return (
            "I can't check the weather, but I can help you send a gift anywhere in "
            "Sri Lanka — cakes, flowers, chocolates, and more."
        )
    if is_impossible_catalog_request(user_text):
        return (
            "I can't deliver a live elephant, but I can help you send a gift instead — "
            "stuffed animals, cakes, or flowers work great."
        )
    lowered = user_text.lower()
    if "add" in lowered and "cart" in lowered:
        return "Added to your cart. You can open the cart drawer anytime to review."
    return (
        "Here are a few options I found for you. Tell me what catches your eye "
        "and I can help you add it to your cart or check delivery."
    )


async def _e2e_generate_content(
    *,
    model: str | None = None,
    messages: list[dict[str, Any]],
    response_schema: Any = None,
    temperature: float = 1.0,
    max_tokens: int = 16384,
    settings: Any = None,
    seed: int | None = 42,
) -> Any:
    """Deterministic, message-aware stand-in for NVIDIA NIM structured completions."""
    _ = (model, temperature, max_tokens, settings, seed)
    schema_name = getattr(response_schema, "__name__", "")
    user_text = _latest_user_text(messages)
    combined = _combined_text(messages)

    if schema_name == "IntentClassification":
        from graphs.nodes.analyze_intent import IntentClassification

        return IntentClassification(intent=_resolve_intent(user_text))  # type: ignore[arg-type]

    if schema_name == "SpecificityRefinement":
        from lib.chat.request_specificity import SpecificityRefinement

        proceed = bool(_SHOPPING_SIGNAL.search(user_text))
        return SpecificityRefinement(
            score=78.0 if proceed else 30.0,
            product_score=0.8 if proceed else 0.2,
            occasion_score=0.7 if proceed else 0.3,
            budget_score=0.6 if proceed else 0.2,
            missing_dimension=None if proceed else "product",
            band="proceed" if proceed else "clarify",
        )

    if schema_name == "AgentPlannerStep":
        from graphs.nodes.agent_loop import AgentPlannerStep

        # Finish after any real prior tool iteration to avoid extra loops / 60s hangs.
        # Match the runtime header (not the static system-prompt docs that also
        # mention "Prior tool iterations").
        if "Prior tool iterations (summarized):" in combined:
            return AgentPlannerStep(action="finish", rationale="catalog facts collected")

        customer = _customer_message_from_planner(user_text)
        if is_off_topic_message(customer) or is_impossible_catalog_request(customer):
            return AgentPlannerStep(
                action="finish",
                refined_intent="general",
                rationale="no catalog tools needed",
            )

        # Delivery-only follow-ups: check_delivery once, then finish next turn.
        if _DELIVERY_SIGNAL.search(customer) and not _SHOPPING_SIGNAL.search(customer):
            city_match = _CITY_SIGNAL.search(customer)
            city = city_match.group("city") if city_match else "Kandy"
            return AgentPlannerStep(
                action="call_tool",
                tool_name=CHECK_DELIVERY_TOOL,
                tool_args={"city": city, "delivery_date": "2026-06-08"},
                refined_intent="discovery",
                rationale="check delivery for date/city follow-up",
            )

        return AgentPlannerStep(
            action="call_tool",
            tool_name=SEARCH_PRODUCTS_TOOL,
            tool_args={"q": _planner_query(user_text)},
            refined_intent="discovery",
            rationale="search catalog",
        )

    if schema_name == "AssistantReply":
        from graphs.nodes.generate_response import AssistantReply

        return AssistantReply(message=_assistant_reply_copy(user_text))

    if schema_name == "MasterFlowAlignment":
        from lib.chat.master_flow import MasterFlowAlignment

        return MasterFlowAlignment(
            decision="proceed",
            confidence=0.9,
            active_flow="carousel_context",
        )

    return {"content": "ok", "role": "assistant"}


_mcp_client: MockMCPHttpClient | None = None
_redis_client: RedisClient | None = None
_genai_client: Any = None


async def _fakeredis_asetup(self: AsyncRedisSaver) -> None:
    """Skip RediSearch index creation; fakeredis lacks FT._LIST."""
    self._key_registry = AsyncCheckpointKeyRegistry(self._redis)


def _apply_e2e_env() -> None:
    get_settings.cache_clear()
    for key, value in _VALID_ENV.items():
        os.environ[key] = value
    os.environ["APP_ENV"] = "development"


def get_e2e_mcp_client() -> MockMCPHttpClient:
    """Return the shared mock MCP client wired into the E2E app."""
    if _mcp_client is None:
        msg = "E2E app not initialized; call create_e2e_app() first"
        raise RuntimeError(msg)
    return _mcp_client


def create_e2e_app() -> FastAPI:
    """Build FastAPI app with in-memory Redis, mock MCP, and a hermetic NIM override."""
    global _mcp_client, _redis_client, _genai_client

    _apply_e2e_env()
    AsyncRedisSaver.asetup = _fakeredis_asetup  # type: ignore[method-assign]

    # Route every production generate_content call to the deterministic stand-in
    # so the E2E server never touches the live NVIDIA NIM endpoint.
    set_override_generate_content(_e2e_generate_content)

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    _redis_client = RedisClient("redis://localhost:6379/0", client=fake)
    _mcp_client = MockMCPHttpClient()
    _genai_client = build_eval_genai_client(None)
    kapruka_service = KaprukaService(_redis_client, _mcp_client)

    application = create_app()

    @asynccontextmanager
    async def e2e_lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.redis = _redis_client
        app.state.mcp_client = _mcp_client
        app.state.kapruka_service = kapruka_service
        app.state.neo4j = None
        app.state.zep = None
        yield

    application.router.lifespan_context = e2e_lifespan

    async def mock_build_deps(request: Any, redis: RedisClient) -> ShoppingGraphDeps:
        return ShoppingGraphDeps(
            kapruka_service=kapruka_service,
            client_ip=client_ip_from_request(request),
            genai_client=_genai_client,
            zep_client=None,
            redis_client=redis,
        )

    chat_route.build_shopping_graph_deps = mock_build_deps

    @application.get("/e2e/mcp-calls", include_in_schema=False)
    async def e2e_mcp_calls() -> dict[str, list[str]]:
        """Expose mock MCP call log for smoke-test assertions (E2E only)."""
        return {"tools": list(_mcp_client.call_log)}

    @application.post("/e2e/mcp-calls/reset", include_in_schema=False)
    async def e2e_mcp_calls_reset() -> dict[str, str]:
        """Clear the mock MCP call log between HybridRAG E2E cases."""
        _mcp_client.call_log.clear()
        return {"status": "ok"}

    @application.post("/e2e/reset", include_in_schema=False)
    async def e2e_reset() -> dict[str, str]:
        """Reset shared E2E state: MCP log, fakeredis checkpoints, hermetic NIM override."""
        _mcp_client.call_log.clear()
        if _redis_client is not None and _redis_client._client is not None:
            await _redis_client._client.flushdb()
        # Re-install override in case another test cleared it; counters are per-call.
        set_override_generate_content(_e2e_generate_content)
        reset_planner = getattr(_genai_client, "reset_planner_state", None)
        if callable(reset_planner):
            reset_planner()
        return {"status": "ok"}

    return application
