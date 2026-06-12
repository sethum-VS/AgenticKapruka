# Changelog

All notable changes to AgenticKapruka are documented here.

## [0.0.12.0] - 2026-06-12

### Added

- Per-turn agent state reset (`_per_turn_agent_reset_fields`) preventing multi-turn tool_trace and clarifying-question leaks
- `lib/chat/delivery_dates.py` for Colombo-grounded delivery date parse, normalize, and past-date rejection
- General intent static welcome path (`build_general_welcome_message`) without Gemini on empty tool traces
- `scripts/verify_chat_loop.py` expanded to 7 scenarios (greeting, gifts, cakes, flowers, product, tracking, delivery)
- Cake search post-filter and planner query-rewrite hints; HTML entity decode on assistant replies
- Chat SSE loading hardening: `htmx:afterRequest` backup and pending bubble cleanup on stream error

### Changed

- Concierge response prompts: warm top 2–3 curation, exact prices, delivery context (replaces utility no-empathy tone)
- `MAX_ITERATIONS` 4 → 3 with force-finish after successful search; 90s chat turn timeout in streaming
- `TrackOrderOutput.amount` coerces MCP Money `{value, currency}` shape to formatted string
- Planner tool arg normalization (`query` → `q`) and canonical dedup in `tool_executor`

### Fixed

- Agent loop exits immediately on MCP tool errors with user-facing tier-1 messages
- Invalid ISO delivery tokens no longer crash turns; bare weekday false positives removed from date parse
- Cake accessory filter falls back to raw results when curation would empty a non-empty catalog
- Stale clarifying questions no longer mask search results when exit reason is not `ask_user`

## [0.0.11.0] - 2026-06-12

### Added

- Bounded ReAct agent loop for discovery and general shopping turns (max 4 Flash planner iterations)
- Shared Kapruka tool executor with Pydantic validation for MCP calls from heuristics and planner
- Product ID regex fast-path that routes `kapruka_get_product` without entering the agent loop
- Planner trace summarization so loop context stays compact while full MCP payloads feed response synthesis
- SSE status events and immediate "Searching catalog…" thinking bubble during multi-step catalog lookups
- `merge_tool_trace` for multi-search carousels and clarifying-question rendering in `generate_response`
- Agent loop debug trace summaries (iteration count, tools, exit reason) without logging full catalog payloads
- Guard-only `analyze_intent` with planner-side discovery vs general refinement (Phase 2 intent collapse)
- Ragas golden cases for multi-step agent planning (anniversary dinner, cakes, thanks, product ID fast-path)

### Changed

- Shopping graph routes discovery/general through HybridRAG hints → agent loop → response; tracking and product ID stay on heuristics
- HybridRAG demoted to soft planner hints only; discovery no longer auto-injects search args from graph context
- Checkout and tracking paths unchanged; Pro model escalation applies to final synthesis only, not planner iterations

### Fixed

- Resilient Gemini fallback, dev debug trace polish, and local UX improvements from prior sprint work

## [0.0.10.0] - 2026-06-10

### Added

- Cross-Encoder reranker service (`ms-marco-MiniLM-L-6-v2`) for post-retrieval GraphRAG relevance scoring
- Hybrid context pipeline reranking with `RERANKER_THRESHOLD` pruning and score-based category/occasion hints
- LoRA fine-tuning dataset generator (`scripts/generate_lora_dataset.py`) with Sri Lankan vernacular templates
- Dynamic LoRA model routing for intent classification and occasion-aware query rewrite via `KAPRUKA_LORA_ENDPOINT_ID`
- Multi-turn eval harness: intent heuristics, LLM judge rubrics, query preprocessor, and shadow dataset
- E2E hybrid RAG fidelity tests with MCP tool alignment and visual/constraint scoring

### Changed

- Discovery search keeps raw user `q` intact; occasion context flows through Gemini rewrite instead of string concatenation
- `retrieve_hybrid_context` wires cross-encoder reranking into the LangGraph retrieval path

### Fixed

- E2E test helper import cleanup for hybrid RAG fidelity suite
