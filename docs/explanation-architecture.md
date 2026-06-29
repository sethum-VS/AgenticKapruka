# Architecture and Design Decisions

This document explains why AgenticKapruka is built the way it is. It is aimed at technical leads and architects evaluating or extending the system.

## Design goals

1. **Grounded responses** — every product fact comes from Kapruka MCP, not LLM imagination
2. **Deterministic checkout** — money-moving flows use a state machine, not open-ended agent loops
3. **Bounded discovery** — shopping search uses a capped ReAct loop (max 3 iterations), not unbounded tool chaining
4. **Curated carousels** — post-search filters remove gift-noise (grocery, accessories) before customers see results
5. **Server-rendered UI** — HTMX partials over a JSON SPA keeps the frontend thin and testable
6. **Graceful degradation** — optional services (Zep, Neo4j) fail open; core shopping still works
7. **Cloud Run fit** — stateless containers, Redis for sessions, 120s Gunicorn timeout for SSE

## Layered architecture

```
Browser (HTMX + Alpine.js + Tailwind — Kapruka Concierge shell)
        │
        ▼
FastAPI (routes, Jinja2 templates, middleware)
        │
        ├── LangGraph Shopping Graph (per chat turn)
        │       ├── load_zep_memory
        │       ├── analyze_intent (guards, specificity gate, pivots)
        │       ├── master_flow (flow-state supervisor on conflict triggers)
        │       ├── route_after_master_flow (lib/chat/routing.py)
        │       ├── retrieve_hybrid_context (Neo4j + Zep)
        │       ├── resolve_delivery_context (city/date preflight)
        │       ├── agent_loop (bounded ReAct planner + curation)
        │       ├── resolve_cart_product → execute_cart_action
        │       ├── call_mcp_tools (tracking fast-path)
        │       ├── generate_response (Flash/Pro)
        │       └── zep_memory_write
        │
        └── LangGraph Checkout Graph (per checkout step)
                └── Deterministic step processors
        │
        ▼
Service clients (Redis, Neo4j, Zep, Kapruka MCP, Vertex AI)
```

## Why LangGraph with two graphs?

A single monolithic agent loop would let the LLM freely navigate checkout — risky for orders involving payment links and PII. Splitting into:

- **Shopping graph** — flexible but bounded tool use for discovery, cart references, and tracking
- **Checkout graph** — fixed step order with validation gates

…gives conversational flexibility where it helps and rigid control where it matters.

## Request specificity gate

Before HybridRAG and the agent loop run, `lib/chat/request_specificity.py` scores discovery messages on product type, occasion, and budget dimensions. Low scores produce a clarifying question via `generate_response` without MCP calls — cheaper and less confusing than searching the entire catalog for "gift ideas."

Heuristic guards (budgeted gift chips, product IDs, topic pivots, proceed-to-checkout) bypass the scorer when context is already actionable.

## Flow-state supervisor (master_flow)

After `analyze_intent`, `lib/chat/master_flow.py` runs a lightweight Flash supervisor when deterministic triggers detect a mismatch between the shopper's message and the active conversation "chapter" (checkout in progress, awaiting delivery date, awaiting clarification, carousel context, delivery resolution, or free discovery).

Pure-Python gates (`should_invoke_master_flow`) fire on conflicts such as:

- Awaiting delivery date but the message does not parse as a date
- Awaiting a specificity dimension but the reply does not address it
- Checkout active while intent classifies as discovery
- Delivery-only inquiries that would incorrectly re-search a stale carousel
- Topic pivots or budget drift in long sessions (configurable turn threshold)

When invoked, Flash returns structured alignment (`proceed`, `clarify`, `pivot`, `redirect`, `checkout_exit`) with optional session patches, context resets, and checkout pause/exit. Patches apply only above `MASTER_FLOW_CONFIDENCE_THRESHOLD` (default 0.75). LLM failure is fail-open — the graph proceeds without patches.

Post-supervisor routing lives in `lib/chat/routing.py` (`route_after_master_flow`). A `master_clarifying_question` short-circuits to `generate_response` without HybridRAG or MCP search.

Toggle with `MASTER_FLOW_ENABLED` (default true). Trace fields appear in debug logs via `lib/debug/trace.py`.

## Bounded agent loop

`graphs/nodes/agent_loop.py` implements a Flash-tier planner that chooses among five MCP tools per iteration (`search_products`, `get_product`, `list_categories`, `check_delivery`, `list_delivery_cities`). The loop caps at three iterations for discovery and two for utility/general paths.

After each search, **product curation** (`lib/chat/product_curation.py`) applies scenario-specific ranking:

- Birthday cake intent → promote cakes, demote accessories
- Chocolate focus → demote floral SKUs
- Flower intent → filter puja, produce, air fresheners
- Recipient hints → gender-aware title ranking
- Budget → hide far-over-budget items, prefer near-budget band
- Gift noise → demote grocery, snacks, low-ticket candy

`lib/chat/search_broadening.py` widens queries on empty first results.

## Why HybridRAG instead of pure vector search?

Kapruka's catalog is organized by occasions and categories, not just semantic similarity. The Neo4j ontology models:

```
(Occasion)-[:OCCASION_TO_CATEGORY]->(Category)-[:CATEGORY_TO_PRODUCT_TYPE]->(ProductType)
```

Vector search finds relevant categories; graph traversal expands to related occasions and product types. Zep preferences layer on top. The combined `hybrid_context` object steers MCP search args — better recall than embedding products alone.

Run `python scripts/bootstrap_neo4j.py` before local QA or evals — without it, hybrid context returns zero products and curation quality drops to MCP-only.

## Carousel product references

`lib/chat/product_reference.py` resolves ordinal ("the first one") and deictic ("that") phrases against the last assistant carousel stored in session state. `resolve_cart_product` → `execute_cart_action` handles cart intent without a full discovery loop.

## Support FAQ and off-topic routing

`lib/chat/support_faq.py` and `lib/chat/off_topic.py` short-circuit to curated copy in `analyze_intent` — no MCP calls, no hallucinated policy text. Support replies link to official Kapruka channels.

## Session and thread rotation

`lib/chat/session.py` signs `ak_session` cookies with HMAC. `rotate_chat_thread()` issues a new LangGraph thread ID while `lib/redis/cart.py` copies cart lines to the new session — fresh conversation, preserved basket.

## Why HTMX instead of React/Vue?

Gift shopping UI is form-heavy (delivery, recipient) but not complex enough to justify a client framework. HTMX swaps HTML partials from the server:

- One language (Python + Jinja2) for UI and logic
- Playwright browser tests render real templates
- No client-side state sync with cart/checkout Redis state

Trade-off: less rich client interactivity than a SPA. Acceptable for a chat-first interface where the server owns state.

## Why Redis Stack (not plain Redis)?

LangGraph's Redis checkpointer requires RediSearch modules. Redis Stack also backs:

- Session cookies and currency preference
- Cart and checkout field state
- MCP read cache (30-minute TTL)
- Per-IP rate limiting counters

## Model routing: Flash vs Pro

| Condition | Model | Rationale |
| --- | --- | --- |
| Default turns | Gemini 2.5 Flash | Low latency, low cost |
| Checkout review | Gemini 2.5 Pro | Higher accuracy for order summaries |
| More than 3 tool calls | Gemini 2.5 Pro | Complex multi-product synthesis |

## Kapruka service resilience

`lib/kapruka/service.py` wraps MCP calls with per-IP rate limiting, 30-minute read cache, and automatic retry when Kapruka returns rate-limit errors.

## Co-purchase recommendations

NetworkX Louvain community detection runs hourly on `CO_PURCHASED_WITH` edges in Neo4j, writing `RECOMMENDS` edges. CPU-only — no GPU in Cloud Run.

An optional cuGraph path (`lib/analytics/cugraph_optional.py`) exists for local GPU dev but is not deployed to production.

## Security boundaries

| Data | Where it lives | Exposure |
| --- | --- | --- |
| Session ID | Signed cookie (`SESSION_SECRET`) | Browser only |
| Cart/checkout PII | Redis (encrypted at rest in Memorystore) | Server only |
| Payment | Kapruka checkout URL | Customer redirected to Kapruka |
| API keys | Secret Manager (prod), `.env` (local) | Never in client |

No CORS middleware — same-origin HTMX only.

## Trade-offs accepted

| Choice | Gain | Cost |
| --- | --- | --- |
| MCP over direct Kapruka REST | Standardized tool interface, cacheable reads | Extra network hop |
| Zep for memory | Cross-session personalization | External dependency, latency on load |
| SSE streaming | Responsive chat feel + status copy | Long-lived connections need Gunicorn timeout tuning |
| Server-rendered HTML | Simple testing and deployment | Less offline/PWA capability |
| Specificity gate | Fewer irrelevant searches | Extra turn for vague openers |
| Post-search curation | Cleaner carousels | Maintenance burden as catalog evolves |

## Related docs

- [README architecture diagram](../README.md#architectural-overview)
- [Design system](../DESIGN.md)
- [HTTP API reference](reference-http-api.md)
- [Environment reference](reference-environment.md)
- [Deploy guide](DEPLOY.md)
