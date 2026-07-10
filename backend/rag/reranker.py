from __future__ import annotations

from dataclasses import replace

from rag.embeddings import tokenize_for_local_search
from rag.vector_store import SearchResult


RERANKER_STRATEGIES = ("none", "local")


def rerank_results(
    query_text: str,
    results: list[SearchResult],
    limit: int,
    strategy: str = "local",
) -> list[SearchResult]:
    """Rerank broad retrieval candidates before building the answer context."""
    if limit <= 0:
        raise ValueError("rerank limit must be greater than 0.")

    if strategy not in RERANKER_STRATEGIES:
        raise ValueError(
            "reranker must be one of: " + ", ".join(RERANKER_STRATEGIES)
        )

    if strategy == "none" or not results:
        return results[:limit]

    query_terms = _unique_terms(query_text)
    base_scores = _normalize_scores([result.score for result in results])
    scored_results = []

    for index, result in enumerate(results):
        rerank_score = _local_rerank_score(
            query_text=query_text,
            query_terms=query_terms,
            result=result,
            base_score=base_scores[index],
        )
        scored_results.append(
            (
                rerank_score,
                result.chunk_index,
                replace(
                    result,
                    score=rerank_score,
                    rerank_score=rerank_score,
                    rerank_strategy=strategy,
                ),
            )
        )

    scored_results.sort(key=lambda item: (-item[0], item[1]))
    return [result for _, _, result in scored_results[:limit]]


def _local_rerank_score(
    query_text: str,
    query_terms: list[str],
    result: SearchResult,
    base_score: float,
) -> float:
    searchable_text = " ".join(
        part
        for part in (result.section_title or "", result.text)
        if part
    )
    text_terms = set(tokenize_for_local_search(searchable_text))
    section_terms = set(tokenize_for_local_search(result.section_title or ""))

    coverage_score = _term_coverage(query_terms, text_terms)
    phrase_score = _phrase_match_score(query_text, result.text)
    section_score = _term_coverage(query_terms, section_terms)

    lexical_score = result.lexical_score or 0.0
    vector_score = result.vector_score or 0.0

    score = (
        0.55 * base_score
        + 0.20 * coverage_score
        + 0.15 * lexical_score
        + 0.05 * vector_score
        + 0.03 * phrase_score
        + 0.02 * section_score
    )
    return round(score, 6)


def _unique_terms(text: str) -> list[str]:
    return list(dict.fromkeys(tokenize_for_local_search(text)))


def _term_coverage(query_terms: list[str], text_terms: set[str]) -> float:
    if not query_terms:
        return 0.0

    matches = sum(1 for term in query_terms if term in text_terms)
    return matches / len(query_terms)


def _phrase_match_score(query_text: str, text: str) -> float:
    query_terms = _unique_terms(query_text)

    if len(query_terms) < 2:
        return 0.0

    lowered_text = text.lower()
    adjacent_pairs = [
        f"{left} {right}"
        for left, right in zip(query_terms, query_terms[1:])
    ]
    matches = sum(1 for pair in adjacent_pairs if pair in lowered_text)
    return matches / len(adjacent_pairs)


def _normalize_scores(scores: list[float]) -> list[float]:
    positive_scores = [max(score, 0.0) for score in scores]
    highest_score = max(positive_scores, default=0.0)

    if highest_score <= 0:
        return [0.0 for _ in positive_scores]

    return [score / highest_score for score in positive_scores]