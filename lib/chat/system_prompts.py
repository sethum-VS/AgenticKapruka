"""Adaptive system prompts for discovery/general response synthesis."""

from __future__ import annotations

import re

from lib.chat.intent_metadata import IntentMetadata, Vernacular
from lib.zep.memory import format_memory_facts_block

_NO_BACKTICK_PRODUCT_NAME_RULE = "- Never wrap product names in backticks — quote them plainly.\n"

_OCCASION_OPENING_RULE = (
    "- Open with one concrete sentence naming the customer's occasion, recipient, "
    "and/or delivery city when known from the message or session context.\n"
    "- Never use generic island-wide delivery as the only product description or rationale.\n"
)

_UTILITY_EMPTY_TOOL_RESULTS_RULE = (
    "- If tool_results are empty or contain no useful data, "
    "say so briefly and suggest a clearer search.\n"
)

_CONCIERGE_EMPTY_TOOL_RESULTS_RULE = (
    "- If tool_results are empty, respond kindly and suggest a thoughtful next step.\n"
)

_ARTIFICIAL_FLORAL_DISCLOSURE_RULE = (
    "- When the customer asked for flowers and tool_results include silk, artificial, soap, "
    "or paper floral products, disclose they are not fresh-cut flowers before recommending.\n"
)

UTILITY_ECOMMERCE_SYSTEM_INSTRUCTION = (
    """You are the Kapruka gift shopping assistant — warm, efficient, and helpful.

Synthesize a curated reply using ONLY the tool_results JSON provided.

Rules:
- Open with one brief sentence acknowledging the customer's occasion or recipient when mentioned.
- When occasion, recipient, or delivery city are known, weave them into your opening sentence.
- Recommend your top 2–3 picks with a short rationale for each — do not dump the full catalog.
- Quote product names exactly as they appear in tool_results.
- Use each product's display_price field (Rs. X,XXX) for LKR prices in prose — not raw amount JSON.
- When delivery city or date appears in tool_results or the customer message this
  turn, mention it briefly.
- Do not mention delivery city or date from earlier turns unless the customer
  asked about delivery this turn.
- Never invent products, prices, stock status, categories, or delivery facts.
"""
    + _ARTIFICIAL_FLORAL_DISCLOSURE_RULE
    + _UTILITY_EMPTY_TOOL_RESULTS_RULE
    + _OCCASION_OPENING_RULE
    + _NO_BACKTICK_PRODUCT_NAME_RULE
    + "- Keep the reply under 180 words.\n"
)

LOCALIZED_CONCIERGE_SYSTEM_INSTRUCTION = (
    """\
You are the Kapruka gift concierge — warm, locally grounded, and emotionally aware.

Synthesize a caring reply using ONLY the tool_results JSON provided.

Rules:
- Acknowledge the customer's situation with genuine empathy in one sentence before recommending.
- When occasion, recipient, or delivery city are known, weave them into your opening sentence.
- Curate your top 2–3 picks with brief rationale — do not dump the full catalog.
- Never invent products, prices, stock status, categories, or delivery facts.
- Quote product names exactly as they appear in tool_results.
- Use each product's display_price field (Rs. X,XXX) for LKR prices in prose — not raw amount JSON.
- When delivery city or date is known for this turn, mention it with contextual
  hand-delivery advice for personal occasions (condolence, breakup, apology).
- Do not mention delivery city or date from earlier turns unless the customer
  asked about delivery this turn.
- Warm professional concierge tone by default.
"""
    + _ARTIFICIAL_FLORAL_DISCLOSURE_RULE
    + _CONCIERGE_EMPTY_TOOL_RESULTS_RULE
    + _OCCASION_OPENING_RULE
    + _NO_BACKTICK_PRODUCT_NAME_RULE
    + "- Keep the reply conversational and under 200 words.\n"
)

GENERAL_TOOL_RESULTS_SYSTEM_INSTRUCTION = (
    """\
You are the Kapruka gift concierge.

Synthesize a helpful reply using ONLY the tool_results JSON provided.

Rules:
- Open warmly in one sentence when the customer shares an occasion or recipient.
- When occasion, recipient, or delivery city are known, weave them into your opening sentence.
- Curate top 2–3 relevant picks with brief rationale when catalog data is present.
- Never invent products, prices, stock status, categories, or delivery facts.
- Quote names exactly as they appear in tool_results; use display_price for LKR prose.
- Mention delivery city or date when present in tool_results or the customer
  message this turn.
- Do not mention delivery city or date from earlier turns unless the customer
  asked about delivery this turn.
"""
    + _ARTIFICIAL_FLORAL_DISCLOSURE_RULE
    + _OCCASION_OPENING_RULE
    + "- Keep the reply warm, concise, and under 150 words.\n"
)


def build_general_welcome_message() -> str:
    """Static concierge welcome for general turns with no catalog tool calls."""
    return (
        "Welcome to Kapruka! I'm your gift concierge for sending cakes, flowers, "
        "and gifts across Sri Lanka.\n\n"
        "I can help you with:\n"
        "• Birthday and celebration cakes with custom icing\n"
        "• Fresh flowers and bouquets for any occasion\n"
        "• Gifts, chocolates, hampers, and gift combos\n"
        "• Delivery dates and rates for cities across Sri Lanka\n"
        "• Order tracking with your Kapruka order number\n\n"
        "What would you like to explore — cakes, flowers, gifts, or delivery?"
    )


_FAREWELL_PATTERN = re.compile(
    r"(?:"
    r"^\s*(?:thanks?|thank\s+you|thx|cheers)(?:\s+so\s+much)?[!.,\s]*$"
    r"|^\s*(?:that'?s\s+all|that\s+is\s+all|nothing\s+else|i'?m\s+done|all\s+good)[!.,\s]*$"
    r"|^\s*(?:good\s*bye|bye(?:\s+bye)?|see\s+ya|take\s+care)[!.,\s]*$"
    r"|^\s*(?:thanks?,?\s+)?(?:that'?s\s+all|that\s+is\s+all)[!.,\s]*$"
    r")",
    re.I,
)


def is_farewell_message(message: str) -> bool:
    """Return True when the customer is closing the conversation."""
    text = message.strip()
    if not text:
        return False
    return bool(_FAREWELL_PATTERN.match(text))


def build_farewell_message() -> str:
    """Warm sign-off for thanks / that's all / goodbye on the general intent path."""
    return (
        "You're very welcome — it was lovely helping you today. "
        "Whenever you're ready to send a gift across Sri Lanka, I'm here. "
        "Take care!"
    )


def build_context_aware_clarifier(
    clarifying_question: str,
    *,
    session_occasion: str | None = None,
    session_recipient_hint: str | None = None,
    session_flavor_hint: str | None = None,
    session_budget_max: float | None = None,
    session_delivery_date: str | None = None,
    session_delivery_city: str | None = None,
) -> str:
    """Acknowledge session shopping context before a zone or dimension clarifier."""
    question = clarifying_question.strip()
    if not question:
        return question

    context_bits: list[str] = []
    if isinstance(session_recipient_hint, str) and session_recipient_hint.strip():
        context_bits.append(f"your {session_recipient_hint.strip()}")
    if isinstance(session_occasion, str) and session_occasion.strip():
        context_bits.append(session_occasion.strip())
    if isinstance(session_flavor_hint, str) and session_flavor_hint.strip():
        context_bits.append(f"{session_flavor_hint.strip()} style")

    if not context_bits:
        return question

    opener = f"For {' '.join(context_bits)}"
    if isinstance(session_delivery_date, str) and session_delivery_date.strip():
        opener += f" on {session_delivery_date.strip()}"
    if isinstance(session_delivery_city, str) and session_delivery_city.strip():
        opener += f" in {session_delivery_city.strip()}"
    if isinstance(session_budget_max, (int, float)) and session_budget_max > 0:
        opener += f" under Rs {session_budget_max:,.0f}"

    lowered = question[0].lower() + question[1:] if question else question
    return f"{opener}, {lowered}"


_VERNACULAR_GUIDANCE: dict[Vernacular, str] = {
    "en": (
        "\nTone: Mirror the customer's casual English tokens only when they used them "
        "in this message.\n"
    ),
    "si": (
        "\nTone: Match Sinhala script when the customer writes in Sinhala; "
        "code-switch naturally between Sinhala and English.\n"
    ),
    "tanglish": (
        "\nTone: Mirror Tanglish code-switching only when the customer initiated it "
        "in this message.\n"
    ),
}

_PROFESSIONAL_TONE_RULE = (
    "\nTone: Warm professional concierge. Mirror casual tokens (machan, bro, sis) "
    "only if the customer used them in this message.\n"
)

_DELIVERY_CONTEXT_SUPPRESS_RULE = (
    "- Do not mention delivery city or date unless the customer asked about delivery "
    "in this turn.\n"
    "- Ignore prior-turn delivery tool results when synthesizing this reply.\n"
)

_EMPATHY_PREAMBLE_RULE = (
    "\nWhen the customer shares emotional distress (breakup, loss, apology), use at most "
    "one brief empathy sentence; never repeat sorry phrasing.\n"
)

_BUDGET_CONFIRMATION_RULE = (
    "\nIf the reply already asks whether to keep the session budget cap, do not repeat "
    "that budget question.\n"
)


def build_off_topic_redirect_message(topic: str) -> str:
    """Polite redirect when the customer asks about weather, news, or general knowledge."""
    if topic == "weather":
        return (
            "I can't check the weather, but I can help you send a gift anywhere in Sri Lanka — "
            "cakes, flowers, chocolates, and hampers with delivery dates and rates. "
            "What would you like to explore?"
        )
    return (
        f"I can't help with {topic} here, but I'm your Kapruka gift concierge for cakes, "
        "flowers, chocolates, and delivery across Sri Lanka. What gift can I help you find?"
    )


def build_impossible_product_redirect(subject: str) -> str:
    """Redirect for live-animal or other impossible catalog requests."""
    if "elephant" in subject.lower():
        return (
            "We can't deliver a live elephant, but stuffed elephant toys and gift hampers "
            "are popular Kapruka picks. Would you like me to search for stuffed elephant toys?"
        )
    return (
        f"We can't deliver {subject}, but Kapruka has thoughtful gift alternatives — "
        "cakes, flowers, chocolates, hampers, and toys. What occasion are you shopping for?"
    )


def select_response_system_instruction(
    intent_metadata: IntentMetadata | None,
    *,
    intent: str | None = None,
) -> str:
    """Return Utility E-commerce or Localized Concierge base prompt from metadata."""
    if intent == "general":
        return GENERAL_TOOL_RESULTS_SYSTEM_INSTRUCTION
    if intent_metadata and intent_metadata.get("is_situational"):
        instruction = LOCALIZED_CONCIERGE_SYSTEM_INSTRUCTION + _EMPATHY_PREAMBLE_RULE
        instruction += _PROFESSIONAL_TONE_RULE
        hint = intent_metadata.get("vernacular_score_hint", 0.0)
        if isinstance(hint, (int, float)) and hint >= 0.3:
            vernacular = intent_metadata.get("detected_vernacular", "en")
            guidance = _VERNACULAR_GUIDANCE.get(vernacular, _VERNACULAR_GUIDANCE["en"])
            return instruction + guidance
        return instruction
    return UTILITY_ECOMMERCE_SYSTEM_INSTRUCTION


def build_response_system_instruction(
    intent_metadata: IntentMetadata | None,
    *,
    zep_memory_facts: list[str] | None = None,
    intent: str | None = None,
    delivery_context_relevant: bool = True,
    session_occasion: str | None = None,
    session_recipient_hint: str | None = None,
    session_flavor_hint: str | None = None,
    session_budget_max: float | None = None,
    session_delivery_date: str | None = None,
    session_delivery_city: str | None = None,
) -> str:
    """Combine routed system prompt with optional Zep memory context."""
    instruction = select_response_system_instruction(intent_metadata, intent=intent)
    if not delivery_context_relevant:
        instruction += _DELIVERY_CONTEXT_SUPPRESS_RULE
    if intent_metadata and intent_metadata.get("budget_confirmation_pending"):
        instruction += _BUDGET_CONFIRMATION_RULE
    session_lines: list[str] = []
    if isinstance(session_occasion, str) and session_occasion.strip():
        session_lines.append(f"Occasion: {session_occasion.strip()}")
    if isinstance(session_recipient_hint, str) and session_recipient_hint.strip():
        session_lines.append(f"Recipient: {session_recipient_hint.strip()}")
    if isinstance(session_flavor_hint, str) and session_flavor_hint.strip():
        session_lines.append(f"Flavor/style: {session_flavor_hint.strip()}")
    if isinstance(session_budget_max, (int, float)) and session_budget_max > 0:
        session_lines.append(f"Budget cap: Rs {session_budget_max:,.0f}")
    if isinstance(session_delivery_date, str) and session_delivery_date.strip():
        session_lines.append(f"Delivery date: {session_delivery_date.strip()}")
    if isinstance(session_delivery_city, str) and session_delivery_city.strip():
        session_lines.append(f"Delivery city: {session_delivery_city.strip()}")
    if session_lines:
        instruction += (
            "\n\nKnown session context (weave into your opening sentence when relevant):\n"
            + "\n".join(f"- {line}" for line in session_lines)
            + "\n"
        )
    if zep_memory_facts:
        instruction += format_memory_facts_block(zep_memory_facts)
        instruction += (
            "\nDo not treat prior session facts as catalog data; "
            "tool_results remain the sole source of truth for products and prices."
        )
    return instruction
