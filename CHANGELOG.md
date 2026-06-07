# Changelog

All notable changes to this project will be documented in this file.

## [0.0.4.0] - 2026-06-08

### Added

- Neo4j ontology batch embedding (`lib/neo4j/embed_ontology.py`) and `scripts/embed_ontology.py` CLI
- Category vector index and similarity search (`lib/neo4j/vector_search.py`)
- HybridRAG `retrieve_hybrid_context` with Neo4j vector search, 2-hop traversal, and Zep preference merge
- Graph-informed MCP discovery filters via `build_discovery_search_args` (`lib/neo4j/hybrid_context.py`)
- UI partials: product card, horizontal carousel, and HTMX category filter chips with `/partials/search` stub
- Integration test for wedding-flowers → Flowers category hybrid context

### Changed

- Ralph loop: optional `RALPH_TIMEOUT`, live plain-text streaming, graceful abort on Ctrl+C

## [0.0.3.0] - 2026-06-08

### Added

- SSE streaming chat at `POST /chat/stream` with LangGraph `astream` updates and HTMX-compatible HTML fragments
- HTMX SSE bridge (`chat-sse.js`) posting form data via `fetch` while `htmx-ext-sse` handles swaps
- Neo4j ontology schema (`lib/neo4j/ontology.py`), category ingest script, and node property enrichment (slug, display_name, kapruka_id)
- Vertex AI `text-embedding-005` client (`lib/embeddings/vertex_embeddings.py`) with batch embed support
- Chat route integration test and assistant message partial with optional product carousel slot

### Fixed

- Commit missing `ontology.py` module required by ingest imports (CI/fresh-clone import failure)
- HMAC-signed `ak_session` cookies prevent client-controlled LangGraph checkpoint thread IDs
- Chat stream setup failures now emit a visible SSE error alert instead of an empty body
- Rate-limit client IP ignores spoofed `X-Forwarded-For` unless behind a trusted proxy

## [0.0.2.0] - 2026-06-08

### Added

- LangGraph shopping orchestration: intent classification, hybrid context retrieval, MCP tool dispatch, and response synthesis nodes
- Compiled shopping `StateGraph` with Redis checkpointer for multi-turn conversation continuity
- Gemini model router escalating from Flash to Pro on checkout review or deep tool chains
- Zep session create/resume with Redis-backed thread mapping and 7-day TTL
- Zep memory load/write nodes injecting prior-session facts into LLM prompts
- Cross-session preference extraction from Zep memory applied to discovery search hints
- Assistant chat bubble partial (`message_assistant.html`) for HTMX swaps

### Fixed

- Clear `tool_results` each graph turn so checkpointed MCP payloads do not leak into later intents
- Zep session mapping uses atomic `SET NX` to avoid duplicate session creation under concurrency
- Zep memory write failures are logged and no longer fail the graph after a response is generated

## [0.0.1.0] - 2026-06-08

### Added

- Kapruka MCP tool wrappers for search, product detail, categories, delivery cities, delivery checks, create order, and track order with typed Pydantic models
- `KaprukaService` facade combining per-IP sliding-window rate limits and 30-minute read cache for MCP tools
- Chat viewport at `/chat` with welcome empty state, suggested prompts, and warm commerce Tailwind styling
- HTMX chat input form posting to `/chat/stream` with user bubble rendering and empty-state removal
- Alpine.js `chatHelpers` for auto-scroll and input refocus after HTMX swaps
- `AgentState` TypedDict schema for LangGraph orchestration (messages reducer, intent, checkout, tool fields)

### Changed

- `create_order` MCP calls skip transient retries to avoid duplicate non-idempotent writes
- Chat message form enforces a 2000-character maximum length

### Fixed

- Replaced `assert` cache guard in `create_order` wrapper with explicit runtime check (safe under `python -O`)
