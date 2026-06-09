# Customer Capabilities Reference

Complete list of what shoppers can do in AgenticKapruka. Use this for QA test planning, stakeholder demos, and support training.

## Conversation intents

| Intent | Trigger examples | System behavior |
| --- | --- | --- |
| `discovery` | "birthday gifts", "cakes under 5000", "show me flowers" | HybridRAG + MCP search + product carousel |
| `checkout` | "checkout", "proceed to order", "buy this" | Seven-step checkout sub-graph |
| `tracking` | "where is order KA-123", "track my delivery" | MCP order status lookup |
| `general` | "hello", "thanks", off-topic | Polite response, redirect to shopping |

## Product discovery

| Capability | Details |
| --- | --- |
| Text search | Full Kapruka catalog via `kapruka_search_products` |
| Category browse | `kapruka_list_categories` with depth control |
| Product detail | By product ID via `kapruka_get_product` |
| Filters | Category, min/max price, in-stock only, sort order |
| Visual display | HTMX product carousel with images, prices, stock badges |
| Currency | LKR (default), USD, GBP, AUD, CAD, EUR |

## Cart

| Action | How |
| --- | --- |
| Add item | Carousel "Add to cart" button or chat request |
| Update quantity | Cart drawer controls |
| Remove item | Cart drawer remove action |
| View total | Cart drawer header badge and line totals |

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

## Order tracking

| Input | Output |
| --- | --- |
| Order number (e.g. `KA-12345678`) | Delivery status card with progress from Kapruka |
| Missing order number | Assistant asks for it |

## Personalization (when Zep + Neo4j available)

| Memory type | Source | Effect |
| --- | --- | --- |
| Currency preference | Session + Zep facts | Prices displayed in preferred currency |
| Occasion interests | Zep facts | Search hints bias toward relevant categories |
| Category affinity | Neo4j HybridRAG | Better category routing for ambiguous queries |
| Cross-session recall | Zep thread memory | Context from prior visits |

## Recommendations

| Type | Mechanism |
| --- | --- |
| Co-purchase | NetworkX Louvain communities on Neo4j `CO_PURCHASED_WITH` edges |
| Category proximity | Graph traversal from matched categories |

## Rate limiting and caching

| Behavior | Customer impact |
| --- | --- |
| Per-IP MCP rate limits | Banner with countdown when exceeded |
| Read cache (30 min) | Faster repeat searches for identical queries |
| Degraded mode | Chat works with reduced features if Neo4j or Zep is down |

## What customers cannot do

- Create a Kapruka account inside chat (session-only, no login)
- Pay inside AgenticKapruka (redirected to Kapruka checkout)
- Cancel or modify orders after finalize (must use Kapruka support)
- Search when MCP is entirely unavailable (graceful error message)

## Related

- [Shopping journey](explanation-shopping-journey.md)
- [HTTP API reference](reference-http-api.md)
