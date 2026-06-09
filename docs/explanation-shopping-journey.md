# The Shopping Journey

This document explains how a customer moves through AgenticKapruka from first message to order confirmation. It is written for product managers, CX teams, and business stakeholders who need to understand the end-to-end experience without reading code.

## Journey map

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Arrive at  │───▶│  Discover    │───▶│  Add to     │───▶│  Checkout    │
│  /chat      │    │  products    │    │  cart       │    │  (7 steps)   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          │                                        │
                          ▼                                        ▼
                   ┌──────────────┐                        ┌──────────────┐
                   │  Return      │                        │  Pay on      │
                   │  visit with  │                        │  Kapruka     │
                   │  memory      │                        │  (external)  │
                   └──────────────┘                        └──────────────┘

        Any time: "Where is order KA-12345?" ──▶ Tracking flow
```

## Phase 1: First visit

When a customer opens `/chat`, the system assigns a session cookie. No account login is required. The chat page loads with an empty conversation and a currency selector (default LKR).

The customer types a message. The message streams back via Server-Sent Events (SSE) — the reply appears token by token as HTML fragments, not as a blocking page reload.

## Phase 2: Discovery

**What the customer does:** Describes what they want — occasion, recipient, budget, or product name.

**What happens behind the scenes:**

1. Gemini classifies the message as `discovery`
2. The query is embedded and matched against Neo4j gift-category ontology (occasions → categories → product types)
3. Zep preferences (if any) merge into search hints — e.g., preferred currency or past interest in flowers
4. Kapruka MCP `search_products` runs with live catalog data
5. Gemini writes a conversational summary; products render as a swipeable carousel

**What the customer sees:** A natural-language reply plus product cards with images, prices in their currency, and stock badges.

## Phase 3: Cart management

Customers add items via carousel actions or by asking in chat. The cart drawer (HTMX partial) updates without leaving the conversation. Cart state lives in Redis per session.

## Phase 4: Checkout

When the customer says "checkout" or "proceed to order," intent switches to `checkout` and a deterministic seven-step flow begins:

| Step | Customer provides | System validates |
| --- | --- | --- |
| Cart | Confirms items and quantities | Stock and totals from Redis |
| Delivery city | City name (autocomplete) | Kapruka delivery city list |
| Delivery date | Preferred date | Kapruka delivery availability API |
| Recipient | Name and phone | Format validation |
| Sender | Name, anonymous flag, gift message | Format validation |
| Review | Confirms summary | Pro-tier Gemini summarizes order |
| Finalize | Confirms payment | Creates Kapruka order, returns checkout URL |

Customers cannot skip steps. If they try to jump ahead, the system holds them at the current step.

**Payment** happens on Kapruka's secure checkout page via the link returned at finalize — AgenticKapruka does not handle card data.

## Phase 5: Return visits

On subsequent visits, Zep loads memory facts from prior sessions: preferred currency, occasions mentioned, product types browsed. The assistant greets with context — "Welcome back! Still looking for anniversary gifts?" — without requiring the customer to repeat themselves.

## Phase 6: Order tracking

At any point, the customer can ask "Where is my order?" or provide an order number. Intent classifies as `tracking`, the system calls `kapruka_track_order`, and a status card renders in chat with delivery progress.

## Edge cases the system handles

| Situation | Behavior |
| --- | --- |
| Product out of stock | Stock badge shown; assistant suggests alternatives |
| Delivery unavailable for city/date | Validation error on the delivery step with retry |
| Rate limit on MCP calls | Banner with countdown; cached reads served when possible |
| Missing Neo4j or Zep | App degrades gracefully — chat works with reduced personalization |
| Off-topic message | Classified as `general`; polite redirect to shopping topics |

## Related docs

- [Tutorial: first conversation](tutorial-first-conversation.md)
- [How to find and order gifts](howto-find-and-order-gifts.md)
- [How to complete checkout](howto-complete-checkout.md)
- [How to track delivery](howto-track-delivery.md)
