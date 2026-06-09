# Tutorial: Your First Conversation

In this tutorial you will start the assistant locally, search for a gift, view products in chat, and add an item to your cart. By the end you will have a working session and understand the basic chat flow.

## What you will need

- AgenticKapruka running locally (see [Developer setup](howto-developer-setup.md) if not yet installed)
- A browser (Chrome, Firefox, or Safari)
- Redis Stack, Neo4j, Zep, and Vertex AI credentials configured in `.env`

## Step 1: Start the server

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

You should see `Application startup complete` in the terminal. If Redis or Neo4j failed to connect, the app still starts but some features will be limited.

## Step 2: Open the chat

Navigate to [http://localhost:8000/chat](http://localhost:8000/chat).

You will see the chat interface with a message input and currency selector. The page redirects from `/` automatically.

## Step 3: Send your first message

Type a gift request and press Enter:

```
I need a birthday cake for my sister in Colombo, budget around 5000 LKR
```

Within a few seconds, the assistant streams a reply. You should see:

1. Your message in a right-aligned bubble
2. An assistant reply on the left
3. A product carousel (if MCP and Neo4j are connected) with cake options, prices, and images

**What just happened:** Your message was classified as `discovery`, HybridRAG retrieved relevant cake categories from Neo4j, Kapruka MCP searched the live catalog, and Gemini wrote a summary grounded in those results.

## Step 4: Change currency (optional)

Use the currency selector in the header to switch to USD. Send another search message. Prices in the carousel should reflect the new currency.

## Step 5: Add to cart

Click **Add to cart** on a product card in the carousel. The cart drawer icon updates with an item count. Open the drawer to confirm the item, quantity, and line total.

## What you built

You now have:

- A live chat session with a signed cookie
- At least one product in your Redis-backed cart
- Evidence that discovery, MCP search, and HTMX rendering work end to end

## Next steps

- [Find and order gifts](howto-find-and-order-gifts.md) — more discovery scenarios
- [Complete checkout](howto-complete-checkout.md) — finish a purchase
- [Track delivery](howto-track-delivery.md) — check order status

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Empty assistant reply | Check `/health` — MCP or Vertex AI may be down |
| No product carousel | Verify `KAPRUKA_MCP_URL` and Neo4j connection in `.env` |
| "Something went wrong" banner | Check terminal logs; Redis must be running with RediSearch |
| Vertex AI auth error | Run `gcloud auth application-default login` |
