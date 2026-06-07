# Changelog

All notable changes to this project will be documented in this file.

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
