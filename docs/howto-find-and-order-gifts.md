# How to Find and Order Gifts

Use this guide to test product discovery scenarios in AgenticKapruka. It assumes the app is running locally or on a deployed environment.

## Prerequisites

- Chat page accessible at `/chat`
- `/health` returns `"status": "healthy"` for `mcp` and `neo4j`

## Search by occasion

1. Open `/chat`
2. Send: `What gifts do you have for a wedding anniversary?`
3. Verify the assistant returns a conversational reply with a product carousel
4. Confirm prices show your session currency

## Search by budget

1. Send: `Show me flowers under 3000 rupees`
2. Verify products appear within the stated budget
3. Switch currency to USD via the header selector and repeat — prices should convert

## Look up a specific product

1. Send a Kapruka product ID if you have one: `Tell me about product cake00ka002034`
2. Verify a single-product detail response (not a carousel)

## Browse categories

1. Send: `What categories do you sell?`
2. Verify category chips or a category list appears
3. Click or ask about a specific category to drill down

## Add multiple items to cart

1. Search for a product and click **Add to cart**
2. Search for a second product and add it
3. Open the cart drawer — both items should appear with correct totals

## Proceed toward checkout

1. With items in cart, send: `Proceed to checkout`
2. Verify the checkout flow begins (delivery city step or cart review)

Continue with [How to complete checkout](howto-complete-checkout.md).

## Verification

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

All services (`redis`, `neo4j`, `neo4j_graphrag`, `zep`, `mcp`) should report `"up"`.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| "I couldn't find products" | MCP down or empty query | Check MCP health; rephrase search |
| Wrong currency | Session default | Use currency selector or `/session/currency` |
| Rate limit banner | Too many MCP calls from same IP | Wait for countdown; results may be cached |
| No carousel, text only | Neo4j down | HybridRAG skipped; MCP search still runs |
