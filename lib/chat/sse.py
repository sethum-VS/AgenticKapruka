"""Server-Sent Events formatting for HTMX chat streaming."""

from __future__ import annotations


def format_sse_event(data: str, *, event: str | None = "message") -> str:
    """Encode a single SSE event with one or more data lines."""
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    payload_lines = data.splitlines()
    if not payload_lines:
        lines.append("data:")
    else:
        lines.extend(f"data: {line}" for line in payload_lines)
    lines.append("")
    return "\n".join(lines) + "\n"


def chunk_text(text: str, *, words_per_chunk: int = 4) -> list[str]:
    """Split text into word chunks for progressive SSE delivery."""
    words = text.split()
    if not words:
        return [text] if text else []
    chunks: list[str] = []
    for index in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[index : index + words_per_chunk]))
    return chunks
