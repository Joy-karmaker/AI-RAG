from __future__ import annotations

import re
from dataclasses import dataclass

from rag.embeddings import tokenize_for_local_search
from rag.vector_store import SearchResult


INSUFFICIENT_CONTEXT_PHRASES = (
    "does not contain enough information",
    "does not provide enough information",
    "not enough information",
    "cannot answer",
    "can't answer",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "chunk",
    "chunks",
    "context",
    "document",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "source",
    "sources",
    "that",
    "the",
    "this",
    "to",
    "used",
    "using",
    "with",
}


@dataclass(frozen=True)
class VerificationResult:
    verification_status: str
    supported: bool
    answer_coverage: float
    evidence_score: float
    cited_chunks: list[int]
    supporting_chunks: list[int]
    missing_citations: list[int]
    notes: str


def verify_grounded_answer(
    question: str,
    answer: str,
    search_results: list[SearchResult],
) -> VerificationResult:
    """Check whether an answer is supported by retrieved chunks."""
    if not answer.strip():
        return VerificationResult(
            verification_status="unsupported",
            supported=False,
            answer_coverage=0.0,
            evidence_score=0.0,
            cited_chunks=[],
            supporting_chunks=[],
            missing_citations=[],
            notes="Answer is empty.",
        )

    if not search_results:
        return VerificationResult(
            verification_status="unsupported",
            supported=False,
            answer_coverage=0.0,
            evidence_score=0.0,
            cited_chunks=[],
            supporting_chunks=[],
            missing_citations=[],
            notes="No retrieved chunks were available for verification.",
        )

    answer_without_sources = _remove_sources_line(answer)
    if _is_insufficient_context_answer(answer_without_sources):
        return VerificationResult(
            verification_status="insufficient_context",
            supported=False,
            answer_coverage=0.0,
            evidence_score=_best_evidence_score(search_results),
            cited_chunks=_extract_cited_chunks(answer),
            supporting_chunks=[],
            missing_citations=[],
            notes="Answer says the retrieved context is insufficient.",
        )

    cited_chunks = _extract_cited_chunks(answer)
    available_chunks = {result.chunk_index for result in search_results}
    missing_citations = sorted(
        chunk_index for chunk_index in cited_chunks if chunk_index not in available_chunks
    )
    supporting_chunks = _rank_supporting_chunks(answer_without_sources, search_results)
    answer_terms = _content_terms(answer_without_sources)
    context_terms = _context_terms(search_results)
    coverage = _coverage(answer_terms, context_terms)
    evidence_score = _best_evidence_score(search_results)
    has_valid_citation = bool(cited_chunks) and not missing_citations

    if coverage >= 0.72 and has_valid_citation and evidence_score >= 0.20:
        status = "supported"
        supported = True
        notes = "Answer terms are well covered by retrieved context and citations are valid."
    elif coverage >= 0.45 and not missing_citations:
        status = "partially_supported"
        supported = False
        if cited_chunks:
            notes = "Answer is partly covered by context but support is not strong enough."
        else:
            notes = "Answer is partly covered by context but does not cite source chunks."
    else:
        status = "unsupported"
        supported = False
        if missing_citations:
            notes = "Answer cites chunks that were not retrieved."
        elif not cited_chunks:
            notes = "Answer does not cite source chunks and support is weak."
        else:
            notes = "Answer contains details that are not well supported by retrieved context."

    return VerificationResult(
        verification_status=status,
        supported=supported,
        answer_coverage=round(coverage, 4),
        evidence_score=round(evidence_score, 4),
        cited_chunks=cited_chunks,
        supporting_chunks=supporting_chunks[:3],
        missing_citations=missing_citations,
        notes=notes,
    )


def _remove_sources_line(answer: str) -> str:
    lines = []
    for line in answer.splitlines():
        if line.strip().lower().startswith(("source:", "sources:")):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _is_insufficient_context_answer(answer: str) -> bool:
    lowered_answer = answer.lower()
    return any(phrase in lowered_answer for phrase in INSUFFICIENT_CONTEXT_PHRASES)


def _extract_cited_chunks(answer: str) -> list[int]:
    source_lines = [
        line
        for line in answer.splitlines()
        if line.strip().lower().startswith(("source:", "sources:"))
    ]
    citation_text = "\n".join(source_lines) if source_lines else answer
    cited = set()

    for match in re.finditer(r"\bchunk(?:s)?\s*#?\s*(\d+)\b", citation_text, re.IGNORECASE):
        cited.add(int(match.group(1)))

    if source_lines:
        for match in re.finditer(r"\b\d+\b", citation_text):
            cited.add(int(match.group(0)))

    return sorted(cited)


def _rank_supporting_chunks(
    answer: str,
    search_results: list[SearchResult],
) -> list[int]:
    answer_terms = set(_content_terms(answer))
    ranked = []

    for result in search_results:
        chunk_terms = set(_content_terms(result.text))
        overlap = len(answer_terms & chunk_terms)
        if overlap > 0:
            ranked.append((overlap, result.score, result.chunk_index))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [chunk_index for _, _, chunk_index in ranked]


def _content_terms(text: str) -> list[str]:
    terms = tokenize_for_local_search(text)
    return [term for term in terms if term not in STOPWORDS and len(term) > 2]


def _context_terms(search_results: list[SearchResult]) -> set[str]:
    terms = set()
    for result in search_results:
        terms.update(_content_terms(result.section_title or ""))
        terms.update(_content_terms(result.text))
    return terms


def _coverage(answer_terms: list[str], context_terms: set[str]) -> float:
    unique_terms = list(dict.fromkeys(answer_terms))
    if not unique_terms:
        return 0.0

    covered = sum(1 for term in unique_terms if term in context_terms)
    return covered / len(unique_terms)


def _best_evidence_score(search_results: list[SearchResult]) -> float:
    return max((result.score for result in search_results), default=0.0)