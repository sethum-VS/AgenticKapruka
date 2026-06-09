# Architecture and Design Decisions

This document explains why AgenticKapruka is built the way it is. It is aimed at technical leads and architects evaluating or extending the system.

## Design goals

1. **Grounded responses** — every product fact comes from Kapruka MCP, not LLM imagination
2. **Deterministic checkout** — money-moving flows use a state machine, not open-ended agent loops
3. **Server-rendered UI** — HTMX partials over a JSON SPA keeps the frontend thin and testable
4. **Graceful degradation** — optional services (Zep, Neo4j) fail open; core shopping still works
5. **Cloud Run fit** — stateless containers, Redis for sessions, 120s Gunicorn timeout for SSE

## Layered architecture

```
Browser (HTMX + Alpine.js + Tailwind)
        │
        ▼
FastAPI (routes, Jinja2 templates, middleware)
        │
        ├── LangGraph Shopping Graph (per chat turn)
        │       ├── Zep memory load/write
        │       ├── Intent classification (Flash)
        │       ├── HybridRAG retrieval (Neo4j + Zep)
        │       ├── MCP tool execution
        │       └── Response generation (Flash/Pro)
        │
        └── LangGraph Checkout Graph (per checkout step)
                └── Deterministic step processors
        │
        ▼
Service clients (Redis, Neo4j, Zep, Kapruka MCP, Vertex AI)
```

## Why LangGraph with two graphs?

A single monolithic agent loop would let the LLM freely navigate checkout — risky for orders involving payment links and PII. Splitting into:

- **Shopping graph** — flexible tool use for discovery and tracking
- **Checkout graph** — fixed step order with validation gates

…gives conversational flexibility where it helps and rigid control where it matters.

## Why HybridRAG instead of pure vector search?

Kapruka's catalog is organized by occasions and categories, not just semantic similarity. The Neo4j ontology models:

```
(Occasion)-[:OCCASION_TO_CATEGORY]->(Category)-[:CATEGORY_TO_PRODUCT_TYPE]->(ProductType)
```

Vector search finds relevant categories; graph traversal expands to related occasions and product types. Zep preferences layer on top. The combined `hybrid_context` object steers MCP search args — better recall than embedding products alone.

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
| SSE streaming | Responsive chat feel | Long-lived connections need Gunicorn timeout tuning |
| Server-rendered HTML | Simple testing and deployment | Less offline/PWA capability |

## Related docs

- [README architecture diagram](../README.md#architectural-overview)
- [HTTP API reference](reference-http-api.md)
- [Environment reference](reference-environment.md)
- [Deploy guide](DEPLOY.md)
