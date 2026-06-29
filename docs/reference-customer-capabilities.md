# Customer Capabilities Reference

Complete list of what shoppers can do in AgenticKapruka. Use this for QA test planning, stakeholder demos, and support training.

## Conversation intents

| Intent | Trigger examples | System behavior |
| --- | --- | --- |
| `discovery` | "birthday cake for mom", "roses under 5000", "anniversary flowers" | Specificity gate → flow supervisor (on conflicts) → HybridRAG → agent loop → curated carousel |
| `cart` | "add the first one to cart", "put that in my cart" | Resolves carousel reference → adds line item |
| `checkout` | "checkout", "proceed to order", "buy this" | Seven-step checkout sub-graph |
| `tracking` | "where is order KA-123", "track my delivery" | MCP order status lookup |
| `general` | "hello", "thanks", off-topic, support FAQ | Curated reply — redirect, policy handoff, or polite decline |

## Request specificity (discovery gate)

Before MCP search runs, a hybrid scorer rates how actionable the request is across product type, occasion, and budget.

| Score band | Customer experience |
| --- | --- |
| **Proceed** | Search runs immediately (explicit product, budget, occasion, or product ID) |
| **Clarify** | Assistant asks one targeted question — product type, occasion/recipient, or budget |
| **Ambiguous** | LLM refines the score; may still clarify instead of searching |

Bypass paths (search without clarifying):

- Kapruka product ID in the message
- Budgeted gift-idea chips (e.g. "gift ideas under Rs 5,000")
- Budget refinement on a prior carousel ("show cheaper options")
- Bare category pivots ("show me cakes") when session context exists
- Proceed-to-checkout phrasing with items already in cart

## Flow-state supervisor

When the shopper's message conflicts with the active session chapter, a Flash supervisor (`master_flow`) may run before catalog search. It is trigger-gated — most turns skip it entirely.

| Trigger | Customer experience |
| --- | --- |
| Delivery-only question with carousel still visible | Delivery answer without a fresh irrelevant product search |
| Checkout active but message is discovery | Checkout pauses or exits (explicit cancel phrases clear checkout) |
| Awaiting clarification but reply off-topic | Targeted clarifying question instead of blind search |
| Long session with budget/recipient drift | Stale carousel context cleared before next discovery search |

Configurable via `MASTER_FLOW_ENABLED`, `MASTER_FLOW_LONG_SESSION_TURNS`, and `MASTER_FLOW_CONFIDENCE_THRESHOLD`. Debug traces log `master_flow_decision` and skip reasons.

## Product discovery

| Capability | Details |
| --- | --- |
| Text search | Full Kapruka catalog via bounded agent loop calling `kapruka_search_products` |
| Category browse | `kapruka_list_categories` with depth control |
| Product detail | By product ID via `kapruka_get_product` |
| Filters | Category, min/max price, in-stock only, sort order |
| Visual display | HTMX product carousel (2-column grid in assistant messages) |
| Currency | LKR (default), USD, GBP, AUD, CAD, EUR |
| Budget refinement | "Under 3000" or "cheaper options" re-ranks carousel toward budget band |
| Topic pivot | Switching category (cakes → flowers) clears sticky occasion context |
| Delivery preflight | City and date resolved before perishable searches when relevant |
| Ambiguous weekdays | Bare "Saturday" or "next Friday" triggers date clarification (Colombo calendar) |

## Product curation (expected carousel quality)

After MCP returns results, curation filters rank and demote off-focus items before display.

| Scenario | Expected outcome |
| --- | --- |
| Birthday + cake intent | Birthday cakes and desserts promoted; cake accessories demoted |
| Chocolate focus | Non-chocolate floral items demoted |
| Flower/fruit intent | Puja items, loose produce, air fresheners filtered |
| Recipient (for her / for him) | Gender-skewed titles ranked toward recipient |
| Gift search | Hampers and combos promoted; grocery, snacks, low-ticket candy demoted |
| Anniversary | Graph-assisted category hints when Neo4j is available |
| Budget stated | Items far above budget hidden; near-budget band preferred |

Without a bootstrapped Neo4j ontology, curation degrades to MCP ranking only.

## Carousel product references

| Phrase | Behavior |
| --- | --- |
| "the first one", "second", "3rd" | Resolves to carousel index from last assistant message |
| "that", "this", "it" | Resolves when exactly one product in context; otherwise asks to clarify |
| Named product from carousel | Fuzzy match against last carousel titles |

## Cart

| Action | How |
| --- | --- |
| Add item | Carousel **Add to cart**, chat reference ("add the first one"), or explicit name |
| Update quantity | Cart drawer panel controls |
| Remove item | Cart drawer remove action |
| View total | Cart drawer trigger badge and line totals |
| New Session | Sidebar **New Session** rotates chat thread; cart copies to new session |

Cart drawer is split into `cart_drawer_trigger.html` (badge + open) and `cart_drawer_panel.html` (line items).

## Checkout steps

| Step | Required fields | Validation source |
| --- | --- | --- |
| `cart` | Items with quantity | Redis cart state |
| `delivery_city` | City name | Kapruka delivery cities API |
| `delivery_date` | Date | Kapruka delivery check API |
| `recipient` | Name, phone | Format rules |
| `sender` | Name, anonymous flag, gift message | Format rules |
| `review` | Confirmation | Pro-tier LLM summary |
| `finalize` | Payment | Kapruka order creation + checkout URL |

Navigation rules:

- Steps must be completed in order
- Back navigation to previous steps is allowed
- Forward skips are rejected
- Side questions during checkout pause the flow; "proceed to checkout" resumes
- Explicit exit phrases (`cancel checkout`, `find something else`, `never mind`) abandon checkout and return to shopping

## Order tracking

| Input | Output |
| --- | --- |
| Order number (e.g. `KA-12345678`) | Delivery status card with progress from Kapruka |
| Missing order number | Assistant asks for it |

## Support FAQ (general intent)

| Topic | Trigger examples | Response |
| --- | --- | --- |
| Returns / refunds | "return policy", "can I get a refund on flowers" | Link to Kapruka shipping/returns policy + support phone |
| Cancellations | "cancel my order", "change my order" | Handoff to Kapruka customer service |
| Quality issues | "flowers arrived wilted", "damaged cake" | Empathetic reply + support contact |
| General support | "customer service", "complaint" | Official channels — not legal advice |

Phone: +94-11-7551111. Policy URL: Kapruka shipping policy page.

## Off-topic and impossible requests

| Type | Examples | Behavior |
| --- | --- | --- |
| Off-topic | Weather, news, sports scores, math homework | Polite redirect back to gift shopping |
| Impossible catalog | "live elephant", "real puppy" | Explains Kapruka cannot fulfill; suggests real gift categories |

## Personalization (when Zep + Neo4j available)

| Memory type | Source | Effect |
| --- | --- | --- |
| Currency preference | Session + Zep facts | Prices displayed in preferred currency |
| Occasion interests | Zep facts | Search hints bias toward relevant categories |
| Category affinity | Neo4j HybridRAG | Better category routing for ambiguous queries |
| Cross-session recall | Zep thread memory | Context from prior visits |
| Session flavor hints | Zep preferences + intent metadata | Carousel focus (cakes, flowers, chocolate) persists until topic pivot |

## Recommendations

| Type | Mechanism |
| --- | --- |
| Co-purchase | NetworkX Louvain communities on Neo4j `CO_PURCHASED_WITH` edges |
| Category proximity | Graph traversal from matched categories |

## Search status UX (SSE)

During multi-iteration agent loops, the composer area shows rotating status strings:

- "Searching our catalog…"
- "Searching Kapruka…"
- "Checking delivery options…"
- "Curating options for your budget…" (when budget is known)
- "Putting together recommendations…"

## Rate limiting and caching

| Behavior | Customer impact |
| --- | --- |
| Per-IP MCP rate limits | Banner with countdown when exceeded; service retries once on rate-limit errors |
| Read cache (30 min) | Faster repeat searches for identical queries |
| Degraded mode | Chat works with reduced personalization if Neo4j or Zep is down |

## Concierge UI

| Element | Behavior |
| --- | --- |
| Sidebar (280px) | New Session, nav placeholders (Settings, Recent Sessions deferred) |
| Welcome state | Suggestion chips for common gift scenarios |
| Composer | Fixed bottom bar, 44px touch targets, spinner during streaming |
| Product grid | 1 column mobile / 2 columns desktop inside assistant bubbles |

Design tokens and layout: [DESIGN.md](../DESIGN.md).

## What customers cannot do

- Create a Kapruka account inside chat (session-only, no login)
- Pay inside AgenticKapruka (redirected to Kapruka checkout)
- Cancel or modify orders after finalize (must use Kapruka support)
- Search when MCP is entirely unavailable (graceful error message)
- Get binding legal advice on returns (FAQ routes to official Kapruka policy)

## Related

- [Shopping journey](explanation-shopping-journey.md)
- [HTTP API reference](reference-http-api.md)
- [Design system](../DESIGN.md)
