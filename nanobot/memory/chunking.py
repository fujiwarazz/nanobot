"""Chunking utilities for memory indexing."""

from __future__ import annotations


def approximate_token_count(text: str) -> int:
    # Roughly 4 chars per token for English; ok for heuristic.
    return max(1, len(text) // 4)


def chunk_text(text: str, target_tokens: int, overlap_tokens: int) -> list[dict[str, object]]:
    if not text.strip():
        return []

    # Split by lines to preserve some structure.
    lines = text.splitlines()
    chunks: list[dict[str, object]] = []
    current: list[str] = []
    current_tokens = 0
    start_line = 1

    for idx, line in enumerate(lines, start=1):
        line_tokens = approximate_token_count(line) + 1
        if current and current_tokens + line_tokens > target_tokens:
            chunk_text = "\n".join(current).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start_line": start_line,
                    "end_line": idx - 1,
                })
            # overlap: keep tail lines until overlap token budget
            if overlap_tokens > 0:
                tail: list[str] = []
                tail_tokens = 0
                for back in reversed(current):
                    tail_tokens += approximate_token_count(back) + 1
                    tail.append(back)
                    if tail_tokens >= overlap_tokens:
                        break
                current = list(reversed(tail))
                current_tokens = sum(approximate_token_count(l) + 1 for l in current)
                start_line = idx - len(current)
            else:
                current = []
                current_tokens = 0
                start_line = idx

        current.append(line)
        current_tokens += line_tokens

    if current:
        chunk_text = "\n".join(current).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "start_line": start_line,
                "end_line": len(lines),
            })

    return chunks
