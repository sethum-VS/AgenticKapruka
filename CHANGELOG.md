# Changelog

All notable changes to AgenticKapruka are documented here.

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
