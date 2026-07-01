# How to Find and Order Gifts

Use this guide to test product discovery scenarios in AgenticKapruka. It assumes the app is running locally or on a deployed environment.

## Prerequisites

- Chat page accessible at `/chat`
- `/health` returns `"status": "healthy"` for `mcp` and `neo4j_graphrag`
- Neo4j bootstrapped: `python scripts/bootstrap_neo4j.py` (required for GraphRAG curation quality)

## Search by occasion

1. Open `/chat`
2. Send: `What gifts do you have for a wedding anniversary?`
3. Verify the assistant returns a conversational reply with a curated product carousel
4. Confirm prices show your session currency
5. Confirm anniversary-appropriate items rank above unrelated grocery or accessories

## Search by budget

1. Send: `Show me flowers under 3000 rupees`
2. Verify products appear within or near the stated budget
3. Switch currency to USD via the header selector and repeat — prices should convert

## Vague gift ideas (clarifying question)

1. Send: `I need gift ideas`
2. **Expected:** Assistant asks a clarifying question (product type, occasion, or budget) — no carousel yet
3. Reply with specifics: `birthday cake for mom under Rs 5,000`
4. **Expected:** Carousel with curated birthday cakes

## Budgeted gift-idea chip

1. Send: `gift ideas under Rs 5,000`
2. **Expected:** Search proceeds without clarifying question; carousel respects budget band

## Budget refinement

1. Complete a search with results in carousel
2. Send: `show me cheaper options` or `under 2000`
3. **Expected:** New or re-ranked carousel closer to the stated budget

## Topic pivot

1. Search for cakes: `birthday cakes`
2. Pivot: `show me flowers instead`
3. **Expected:** Flower carousel; prior cake occasion context does not pollute results

## Carousel product reference

1. Run any search that returns a carousel
2. Send: `add the first one to my cart`
3. **Expected:** Cart drawer badge increments; item matches first carousel product
4. Optional: `add the second one` — second distinct item in cart

## Look up a specific product

1. Send a Kapruka product ID if you have one: `Tell me about product cake00ka002034`
2. Verify a single-product detail response (not a carousel)

## Browse categories

1. Send: `What categories do you sell?`
2. Verify category chips or a category list appears
3. Click or ask about a specific category to drill down

## Support FAQ

1. Send: `What is your return policy?`
2. **Expected:** Reply with Kapruka policy link and support phone — no invented legal text

## Off-topic redirect

1. Send: `What's the weather in Colombo?`
2. **Expected:** Polite redirect back to gift shopping — no weather answer

## Add multiple items to cart

1. Search for a product and click **Add to cart**
2. Search for a second product and add it
3. Open the cart drawer — both items should appear with correct totals

## New Session (cart preserved)

1. Add an item to cart
2. Click sidebar **New Session**
3. **Expected:** Welcome state returns; cart drawer still shows the item

## Proceed toward checkout

1. With items in cart, send: `Proceed to checkout`
2. Verify the checkout flow begins (delivery city step or cart review)

Continue with [How to complete checkout](howto-complete-checkout.md).

## Verification

```bash
curl -s http://localhost:8080/health | python -m json.tool
```

All services (`redis`, `neo4j`, `neo4j_graphrag`, `zep`, `mcp`) should report `"up"`.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| "I couldn't find products" | MCP down or empty query | Check MCP health; rephrase search |
| Clarifying question on specific query | Specificity scorer needs more context | Add product type, occasion, or budget |
| Wrong currency | Session default | Use currency selector or `/session/currency` |
| Rate limit banner | Too many MCP calls from same IP | Wait for countdown; results may be cached |
| No carousel, text only | Neo4j down or not bootstrapped | Run `bootstrap_neo4j.py`; check `neo4j_graphrag` health |
| Grocery items in gift carousel | Neo4j/curation degraded | Bootstrap Neo4j; verify hybrid context returns products |
| "add the first one" fails | No prior carousel in session | Run a search first |
