# Product Overview

AgenticKapruka is a conversational shopping assistant built for Kapruka, Sri Lanka's largest e-commerce platform. Instead of navigating category pages and filters, customers describe what they need in plain language — "birthday cake for my mom in Colombo under 5000 rupees" — and the assistant searches live Kapruka inventory, remembers preferences, and guides them through checkout.

## The problem

Traditional e-commerce works well when the customer already knows what to buy. Gift shopping is different. Buyers often start with an occasion and a recipient, not a product SKU. They need guidance across thousands of categories, delivery constraints, and currency options. A static search box leaves too much work on the customer.

## What AgenticKapruka delivers

| Stakeholder outcome | How the product achieves it |
| --- | --- |
| Higher gift conversion | Occasion-aware search via Neo4j GraphRAG narrows the catalog before MCP queries run |
| Repeat purchase lift | Zep memory recalls currency, occasions, and past interests across sessions |
| Lower support load | Order tracking, delivery validation, and support FAQ handoff happen inside chat |
| Brand-safe answers | Responses are grounded in live Kapruka MCP tool results — the LLM cannot invent prices, stock, or return policy |
| International buyers | Six currencies (LKR, USD, GBP, AUD, CAD, EUR) with session-level preference |

## Core capabilities

1. **Discovery** — natural-language product search with curated visual carousels in chat
2. **Clarifying questions** — vague gift requests get a focused follow-up before search runs
3. **Personalization** — hybrid memory from Zep facts and Neo4j category graphs
4. **Cart references** — "add the first one" resolves against the last carousel
5. **Checkout** — seven-step guided flow from cart to Kapruka payment link
6. **Tracking** — order status lookup by order number
7. **Support FAQ** — returns, refunds, and quality issues route to official Kapruka channels
8. **Recommendations** — co-purchase community detection via NetworkX on Neo4j
9. **Concierge UI** — sidebar workspace with New Session, suggestion chips, and design tokens

## What it is not

- Not a replacement for kapruka.com — it is an assistant layer on top of Kapruka's existing catalog and order APIs
- Not a general-purpose chatbot — intents are scoped to shopping, checkout, tracking, and Kapruka support FAQ; off-topic requests are redirected
- Not offline-capable — live MCP calls are required for product and order data

## Deployment model

Production runs on Google Cloud Run with Memorystore Redis (VPC), Neo4j AuraDB, Zep Cloud, and Vertex AI. The Kapruka MCP server is reached over the public internet.

## Related docs

- [Shopping journey](explanation-shopping-journey.md) — step-by-step customer flows
- [Architecture](explanation-architecture.md) — technical design for engineering leads
- [Customer capabilities reference](reference-customer-capabilities.md) — complete feature list
