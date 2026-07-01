"""Customer-facing status strings for chat SSE and agent nodes."""

SENDING = "Sending…"
SEARCHING_KAPRUKA = "Searching Kapruka…"
SEARCHING_CATALOG = "Searching our catalog…"
CHECKING_DELIVERY = "Checking delivery options…"
CURATING_FOR_BUDGET = "Curating options for your budget…"
PUTTING_TOGETHER_RECOMMENDATIONS = "Putting together recommendations…"

_LONG_SEARCH_STATUS_MESSAGES: tuple[str, ...] = (
    SEARCHING_CATALOG,
    SEARCHING_KAPRUKA,
    CHECKING_DELIVERY,
    CURATING_FOR_BUDGET,
    PUTTING_TOGETHER_RECOMMENDATIONS,
)


def long_search_status_message(*, iteration: int, has_budget: bool = False) -> str:
    """Rotate richer status copy during multi-iteration agent-loop searches."""
    if has_budget and iteration >= 2:
        return CURATING_FOR_BUDGET
    index = min(max(iteration, 0), len(_LONG_SEARCH_STATUS_MESSAGES) - 1)
    return _LONG_SEARCH_STATUS_MESSAGES[index]
