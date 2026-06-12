"""Re-export intent heuristics for evals (implementation lives in lib.chat)."""

from lib.chat.intent_heuristics import infer_intent_from_message

__all__ = ["infer_intent_from_message"]
