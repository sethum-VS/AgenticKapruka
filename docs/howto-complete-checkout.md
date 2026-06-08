# How to Complete Checkout

This guide walks through the seven-step checkout flow from cart review to Kapruka payment link.

## Prerequisites

- At least one item in the cart (see [Find and order gifts](howto-find-and-order-gifts.md))
- Kapruka MCP delivery tools available (`mcp` service up in `/health`)

## Steps

### 1. Start checkout

In chat, send:

```
Proceed to checkout
```

The assistant routes to the checkout sub-graph starting at the **cart** step.

### 2. Confirm cart

Review items and quantities. Adjust via the cart drawer if needed. Advance when ready.

### 3. Delivery city

Enter a delivery city. An autocomplete suggests valid Kapruka delivery cities as you type.

Expected: city accepted or validation error if undeliverable.

### 4. Delivery date

Select or type a delivery date. The system calls `kapruka_check_delivery` to validate availability and fees.

Expected: available dates confirmed; unavailable dates show an error with retry.

### 5. Recipient details

Provide recipient name and phone number. Invalid formats show inline field errors without losing other data.

### 6. Sender details

Provide sender name, choose anonymous delivery if desired, and optionally add a gift message.

### 7. Review

Gemini Pro summarizes the full order: items, delivery, recipient, sender. Read carefully and confirm.

### 8. Finalize and pay

On confirmation, the system calls `kapruka_create_order` and returns a secure Kapruka checkout URL.

Click the link to complete payment on Kapruka.com. AgenticKapruka does not process card data.

## Verification

After finalize, you should receive:

- An order reference number in chat
- A clickable checkout/payment URL
- An expiration time for the payment link

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Cannot advance past delivery | Check city and date against Kapruka delivery coverage |
| Review summary wrong | Go back to the relevant step and correct fields |
| No payment link | Check MCP connectivity; review server logs for `create_order` errors |
| Step skip rejected | Complete steps in order — the state machine blocks forward jumps |

## Related

- [Shopping journey explanation](explanation-shopping-journey.md)
- [Customer capabilities](reference-customer-capabilities.md)
