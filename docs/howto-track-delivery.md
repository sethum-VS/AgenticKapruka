# How to Track a Delivery

Use this guide to test order tracking inside the chat interface.

## Prerequisites

- A valid Kapruka order number (from a test or production order)
- `mcp` service up in `/health`

## Steps

### 1. Open chat

Navigate to `/chat`.

### 2. Ask about your order

Send a message with your order number:

```
Where is order KA-12345678?
```

Or without a number:

```
Track my order
```

The assistant will ask for the order number if not provided.

### 3. Read the status card

The response includes a tracking status card with delivery progress from Kapruka's live order API.

## Verification

- Intent should classify as `tracking` (no product carousel)
- Status data matches Kapruka.com order tracking for the same number
- Invalid order numbers produce a polite error, not fabricated status

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Order not found" | Verify order number format and that the order exists in Kapruka |
| Generic error | Check MCP health endpoint |
| Assistant searches products instead | Rephrase with "track" or "where is my order" |

## Related

- [Shopping journey](explanation-shopping-journey.md)
- [Customer capabilities](reference-customer-capabilities.md)
