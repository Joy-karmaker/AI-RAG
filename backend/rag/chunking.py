from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start: int
    end: int
    page: int | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class TextBlock:
    text: str
    start: int
    end: int
    page: int | None
    section_title: str | None
    is_heading: bool = False


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    strategy: str = "character",
) -> list[TextChunk]:
    """Split text into overlapping chunks for retrieval."""
    _validate_chunk_settings(chunk_size, overlap)

    clean_text = text.strip()
    if not clean_text:
        return []

    if strategy == "character":
        return _chunk_by_character(clean_text, chunk_size, overlap)

    if strategy == "paragraph":
        return _chunk_blocks(
            _parse_document_blocks(clean_text, detect_headings=False),
            chunk_size=chunk_size,
            overlap=overlap,
        )

    if strategy == "heading":
        return _chunk_blocks(
            _parse_document_blocks(clean_text, detect_headings=True),
            chunk_size=chunk_size,
            overlap=overlap,
        )

    if strategy == "page":
        return _chunk_blocks(
            _parse_document_blocks(clean_text, detect_headings=True),
            chunk_size=chunk_size,
            overlap=overlap,
            keep_pages_separate=True,
        )

    raise ValueError(
        "strategy must be one of: character, paragraph, heading, page."
    )


def _chunk_by_character(
    clean_text: str,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
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


def _parse_document_blocks(
    text: str,
    detect_headings: bool,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    current_page: int | None = None
    current_section: str | None = None

    for match in re.finditer(r"\S(?:.*\S)?", text):
        line = match.group(0).strip()
        page_number = _page_marker_number(line)

        if page_number is not None:
            current_page = page_number
            continue

        if detect_headings and _looks_like_heading(line):
            current_section = line
            blocks.append(
                TextBlock(
                    text=line,
                    start=match.start(),
                    end=match.end(),
                    page=current_page,
                    section_title=current_section,
                    is_heading=True,
                )
            )
            continue

        blocks.append(
            TextBlock(
                text=line,
                start=match.start(),
                end=match.end(),
                page=current_page,
                section_title=current_section if detect_headings else None,
            )
        )

    return blocks


def _chunk_blocks(
    blocks: list[TextBlock],
    chunk_size: int,
    overlap: int,
    keep_pages_separate: bool = False,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    current_blocks: list[TextBlock] = []
    current_length = 0

    for block in blocks:
        block_length = len(block.text)
        should_start_new_page_chunk = (
            keep_pages_separate
            and current_blocks
            and block.page != current_blocks[-1].page
        )
        should_start_new_size_chunk = (
            current_blocks
            and current_length + 2 + block_length > chunk_size
        )
        should_start_new_section_chunk = (
            block.is_heading
            and current_blocks
            and block.section_title != current_blocks[-1].section_title
        )

        if (
            should_start_new_page_chunk
            or should_start_new_size_chunk
            or should_start_new_section_chunk
        ):
            _append_structured_chunk(chunks, current_blocks)
            current_blocks = _overlap_blocks(current_blocks, overlap)
            current_length = _blocks_text_length(current_blocks)

            if should_start_new_page_chunk or should_start_new_section_chunk:
                current_blocks = []
                current_length = 0

        if block_length > chunk_size:
            if current_blocks:
                _append_structured_chunk(chunks, current_blocks)
                current_blocks = []
                current_length = 0

            for chunk in _chunk_by_character(block.text, chunk_size, overlap):
                chunks.append(
                    TextChunk(
                        index=len(chunks) + 1,
                        text=_prefix_section_title(
                            chunk.text,
                            block.section_title,
                        ),
                        start=block.start + chunk.start,
                        end=block.start + chunk.end,
                        page=block.page,
                        section_title=block.section_title,
                    )
                )
            continue

        current_blocks.append(block)
        current_length = _blocks_text_length(current_blocks)

    if current_blocks:
        _append_structured_chunk(chunks, current_blocks)

    return chunks


def _append_structured_chunk(
    chunks: list[TextChunk],
    blocks: list[TextBlock],
) -> None:
    if not blocks:
        return

    section_title = _dominant_section_title(blocks)
    text = "\n".join(block.text for block in blocks).strip()
    text = _prefix_section_title(text, section_title)

    if not text:
        return

    chunks.append(
        TextChunk(
            index=len(chunks) + 1,
            text=text,
            start=blocks[0].start,
            end=blocks[-1].end,
            page=blocks[0].page,
            section_title=section_title,
        )
    )


def _overlap_blocks(blocks: list[TextBlock], overlap: int) -> list[TextBlock]:
    if overlap <= 0:
        return []

    overlapped: list[TextBlock] = []
    total_length = 0

    for block in reversed(blocks):
        overlapped.insert(0, block)
        total_length += len(block.text) + 2

        if total_length >= overlap:
            break

    return overlapped


def _blocks_text_length(blocks: list[TextBlock]) -> int:
    if not blocks:
        return 0

    return sum(len(block.text) for block in blocks) + (2 * (len(blocks) - 1))


def _dominant_section_title(blocks: list[TextBlock]) -> str | None:
    for block in blocks:
        if block.section_title:
            return block.section_title

    return None


def _prefix_section_title(text: str, section_title: str | None) -> str:
    if not section_title:
        return text

    if text.lower().startswith(section_title.lower()):
        return text

    return f"{section_title}\n{text}"


def _page_marker_number(line: str) -> int | None:
    match = re.fullmatch(r"-{3,}\s*Page\s+(\d+)\s*-{3,}", line, flags=re.I)

    if not match:
        return None

    return int(match.group(1))


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip(" :")

    if not stripped or len(stripped) > 80:
        return False

    if stripped.startswith(("•", "-", "*")):
        return False

    if ":" in stripped:
        return False

    letters = [character for character in stripped if character.isalpha()]

    if len(letters) < 3:
        return False

    uppercase_ratio = sum(1 for character in letters if character.isupper()) / len(letters)
    if uppercase_ratio >= 0.8:
        return True

    markdown_heading = re.fullmatch(r"#{1,6}\s+.+", line)
    return bool(markdown_heading)


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
