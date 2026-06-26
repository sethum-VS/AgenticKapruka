# The Shopping Journey

This document explains how a customer moves through AgenticKapruka from first message to order confirmation. It is written for product managers, CX teams, and business stakeholders who need to understand the end-to-end experience without reading code.

## Journey map

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Arrive at  │───▶│  Discover    │───▶│  Add to     │───▶│  Checkout    │
│  /chat      │    │  products    │    │  cart       │    │  (7 steps)   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
       │                  │                                        │
       │                  ▼                                        ▼
       │           ┌──────────────┐                        ┌──────────────┐
       │           │  Clarify or  │                        │  Pay on      │
       │           │  curate      │                        │  Kapruka     │
       │           └──────────────┘                        │  (external)  │
       │                                                   └──────────────┘
       ▼
┌──────────────┐
│  New Session │── cart preserved, fresh conversation thread
└──────────────┘

Any time: "Where is order KA-12345?" ──▶ Tracking flow
Any time: "return policy?" ──▶ Support FAQ handoff
```

## Phase 1: First visit

When a customer opens `/chat`, the system assigns a signed session cookie (`ak_session`). No account login is required.

The **Kapruka Concierge** workspace loads:

- Purple sidebar with **New Session** (gold CTA)
- Centered chat column with welcome state and suggestion chips
- Fixed composer at the bottom
- Currency selector in the header (default LKR)

The customer types a message. The reply streams via Server-Sent Events (SSE) — HTML fragments appear progressively, with status copy ("Searching our catalog…") during longer searches.

## Phase 2: Discovery

**What the customer does:** Describes what they want — occasion, recipient, budget, or product name.

**What happens behind the scenes:**

1. Routing guards classify intent (`discovery`, `cart`, `checkout`, `tracking`, or `general`)
2. For vague gift queries, a **specificity scorer** may ask a clarifying question instead of searching
3. HybridRAG embeds the query and matches Neo4j gift-category ontology
4. Delivery city/date may be resolved or clarified before perishable searches
5. A bounded **agent loop** (up to 3 planner iterations) calls Kapruka MCP tools
6. **Product curation** filters rank results (birthday cakes, chocolate focus, recipient, budget band, gift-noise removal)
7. Gemini writes a conversational summary; products render in a 2-column carousel grid

**What the customer sees:** A natural-language reply plus product cards with images, prices in their currency, and stock badges.

### Clarifying instead of searching

If the customer says only "gift ideas" without occasion, product type, or budget, the assistant asks one focused question — for example, "What type of gift — flowers, cake, voucher, or hamper?"

Budgeted chips like "gift ideas under Rs 5,000" proceed directly to search.

### Topic pivots

When the customer switches category mid-conversation (cakes → flowers), sticky occasion context clears so the new search is not polluted by the prior topic.

## Phase 3: Cart management

Customers add items via:

- Carousel **Add to cart** buttons
- Ordinal references: "add the first one"
- Deictic references: "put that in my cart" (when one product is in context)

The cart drawer updates via HTMX without leaving the conversation. Cart state lives in Redis per session thread.

**New Session:** Clicking sidebar **New Session** starts a fresh chat thread but copies cart items to the new session so the customer does not lose their basket.

## Phase 4: Checkout

When the customer says "checkout" or "proceed to order," intent switches to `checkout` and a deterministic seven-step flow begins:

| Step | Customer provides | System validates |
| --- | --- | --- |
| Cart | Confirms items and quantities | Stock and totals from Redis |
| Delivery city | City name (autocomplete) | Kapruka delivery city list |
| Delivery date | Preferred date | Kapruka delivery availability API; ambiguous weekdays clarified |
| Recipient | Name and phone | Format validation |
| Sender | Name, anonymous flag, gift message | Format validation |
| Review | Confirms summary | Pro-tier Gemini summarizes order |
| Finalize | Confirms payment | Creates Kapruka order, returns checkout URL |

Customers cannot skip steps. If they try to jump ahead, the system holds them at the current step.

**Payment** happens on Kapruka's secure checkout page via the link returned at finalize — AgenticKapruka does not handle card data.

## Phase 5: Return visits

On subsequent visits, Zep loads memory facts from prior sessions: preferred currency, occasions mentioned, product types browsed. The assistant can greet with context without requiring the customer to repeat themselves.

Session flavor hints (last focus: cakes, flowers, chocolate) persist until the customer pivots topic.

## Phase 6: Order tracking

At any point, the customer can ask "Where is my order?" or provide an order number. Intent classifies as `tracking`, the system calls `kapruka_track_order`, and a status card renders in chat with delivery progress.

## Phase 7: Support and off-topic

| Customer says | Experience |
| --- | --- |
| "What is your return policy?" | Curated FAQ with Kapruka policy link and support phone |
| "My flowers arrived damaged" | Empathetic handoff to customer service |
| "What's the weather in Colombo?" | Polite redirect to gift shopping |
| "I want a live elephant" | Explains impossible request; suggests real categories |

## Edge cases the system handles

| Situation | Behavior |
| --- | --- |
| Vague "gift ideas" | Clarifying question before search |
| Product out of stock | Stock badge shown; assistant suggests alternatives |
| Delivery unavailable for city/date | Validation error on the delivery step with retry |
| Ambiguous weekday ("Saturday") | Asks which Saturday (this week vs next) |
| Rate limit on MCP calls | Banner with countdown; automatic retry; cached reads when possible |
| Missing Neo4j or Zep | App degrades gracefully — MCP search still runs, less curation |
| Off-topic message | Classified as `general`; redirect to shopping |
| Carousel reference with no context | Asks customer to pick from the last carousel |

## Related docs

- [Tutorial: first conversation](tutorial-first-conversation.md)
- [How to find and order gifts](howto-find-and-order-gifts.md)
- [How to complete checkout](howto-complete-checkout.md)
- [How to track delivery](howto-track-delivery.md)
- [Customer capabilities reference](reference-customer-capabilities.md)
