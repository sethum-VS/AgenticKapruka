# Changelog

All notable changes to AgenticKapruka are documented here.

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
