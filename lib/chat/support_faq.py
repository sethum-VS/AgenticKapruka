"""Support and policy FAQ responses for returns, refunds, and quality issues."""

from __future__ import annotations

import re
from typing import Literal

SupportTopic = Literal["returns", "quality", "cancellation", "general_support"]

KAPRUKA_RETURNS_POLICY_URL = (
    "https://www.kapruka.com/contactUs/shippingPolicyDelivery.jsp"
)
KAPRUKA_SUPPORT_PHONE = "+94-11-7551111"

_RETURN_REFUND_RE = re.compile(
    r"\b(?:return(?:s|ed|ing)?|refund(?:s|ed|ing)?|exchange|money\s+back)\b",
    re.I,
)
_CANCELLATION_RE = re.compile(
    r"\b(?:cancel(?:lation|led|ing)?|change\s+my\s+order|modify\s+(?:my\s+)?order)\b",
    re.I,
)
_POLICY_RE = re.compile(
    r"\b(?:return\s+policy|refund\s+policy|cancellation\s+policy|your\s+policy)\b",
    re.I,
)
_QUALITY_RE = re.compile(
    r"\b(?:wilted|damaged|defective|poor\s+quality|bad\s+quality|not\s+fresh|"
    r"arrived\s+(?:dead|broken|spoiled)|quality\s+issue)\b",
    re.I,
)
_SUPPORT_CONTEXT_RE = re.compile(
    r"\b(?:support|customer\s+service|complain|complaint|help\s+with\s+my\s+order)\b",
    re.I,
)


def is_support_question(message: str) -> bool:
    """True when the customer asks about returns, refunds, cancellations, or quality."""
    stripped = message.strip()
    if not stripped:
        return False
    if _POLICY_RE.search(stripped):
        return True
    if _QUALITY_RE.search(stripped):
        return True
    if _RETURN_REFUND_RE.search(stripped) and re.search(
        r"\b(?:policy|flowers?|cake|gift|order|item|product)\b",
        stripped,
        re.I,
    ):
        return True
    if _CANCELLATION_RE.search(stripped):
        return True
    if _RETURN_REFUND_RE.search(stripped) and _SUPPORT_CONTEXT_RE.search(stripped):
        return True
    return bool(
        _RETURN_REFUND_RE.search(stripped)
        and re.search(r"\b(?:if|when|how)\b", stripped, re.I)
    )


def classify_support_topic(message: str) -> SupportTopic:
    """Classify the support FAQ topic for tailored handoff copy."""
    stripped = message.strip()
    if _QUALITY_RE.search(stripped):
        return "quality"
    if _CANCELLATION_RE.search(stripped):
        return "cancellation"
    if _RETURN_REFUND_RE.search(stripped) or _POLICY_RE.search(stripped):
        return "returns"
    return "general_support"


def build_support_faq_reply(message: str) -> str:
    """Curated Kapruka support handoff — not legal advice; routes to official channels."""
    topic = classify_support_topic(message)
    policy_link = KAPRUKA_RETURNS_POLICY_URL
    phone = KAPRUKA_SUPPORT_PHONE

    quality_block = (
        "For perishable gifts such as fresh flowers or cakes, contact Kapruka support "
        f"as soon as possible with your order number and clear photos of the issue. "
        f"Quality concerns are typically reviewed within a short window after delivery."
    )
    returns_block = (
        "Kapruka handles returns and refunds through their support team — eligibility "
        "depends on the product type and whether the item arrived damaged, defective, "
        "or incorrect. Change-of-mind returns are generally not accepted for perishables."
    )
    cancellation_block = (
        "Orders can only be cancelled before processing begins. After an order is "
        "finalized, changes or cancellations must go through Kapruka support — this "
        "assistant cannot modify completed orders."
    )

    if topic == "quality":
        lead = quality_block
    elif topic == "cancellation":
        lead = cancellation_block
    elif topic == "returns":
        lead = returns_block
    else:
        lead = (
            f"{returns_block} {quality_block}"
        )

    return (
        f"{lead}\n\n"
        f"Please reach Kapruka support at {phone} or review their shipping and returns "
        f"policy: {policy_link}\n\n"
        "I'm a shopping assistant only — I can't process refunds or change finalized "
        "orders on your behalf. If you still need gift ideas or help placing a new "
        "order, I'm happy to help."
    )
