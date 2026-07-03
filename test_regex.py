import re

_IMPOSSIBLE_LIVE_ANIMAL = re.compile(
    r"\b(?:live|real)\s+(?:elephant|tiger|lion|whale|dolphin|puppy|kitten|snake)\b",
    re.I,
)

msg = "Can you deliver a live elephant?"
match = _IMPOSSIBLE_LIVE_ANIMAL.search(msg)
print("MATCHED:" if match else "NOT MATCHED")
