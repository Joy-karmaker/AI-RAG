from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start: int
    end: int


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[TextChunk]:
    """Split text into overlapping chunks for retrieval."""
    _validate_chunk_settings(chunk_size, overlap)

    clean_text = text.strip()
    if not clean_text:
        return []

    chunks: list[TextChunk] = []
    start = 0

    while start < len(clean_text):
        end = _choose_chunk_end(clean_text, start, chunk_size)
        chunk = clean_text[start:end].strip()

        if chunk:
            chunks.append(
                TextChunk(
                    index=len(chunks) + 1,
                    text=chunk,
                    start=start,
                    end=end,
                )
            )

        if end >= len(clean_text):
            break

        start = _choose_next_start(clean_text, end, overlap)

    return chunks


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")


def _choose_chunk_end(text: str, start: int, chunk_size: int) -> int:
    hard_end = min(start + chunk_size, len(text))

    if hard_end >= len(text):
        return len(text)

    minimum_useful_end = start + int(chunk_size * 0.6)
    boundary_candidates = [
        text.rfind("\n\n", start, hard_end),
        text.rfind(". ", start, hard_end),
        text.rfind(" ", start, hard_end),
    ]
    best_boundary = max(boundary_candidates)

    if best_boundary >= minimum_useful_end:
        return best_boundary + 1

    return hard_end


def _choose_next_start(text: str, end: int, overlap: int) -> int:
    start = max(0, end - overlap)

    if start == 0:
        return start

    if text[start - 1].isspace():
        return _skip_whitespace(text, start)

    search_limit = min(len(text), start + 40)

    for position in range(start, search_limit):
        if text[position].isspace():
            return _skip_whitespace(text, position + 1)

    return start


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1

    return position
