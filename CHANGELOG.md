# Changelog

All notable changes to this project will be documented in this file.

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
