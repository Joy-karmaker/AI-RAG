from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import dataclass

from rag.chunking import TextChunk
from rag.embeddings import EmbeddingResult, tokenize_for_local_search


DEFAULT_COLLECTION_NAME = "ai_rag_chunks"
HYBRID_LEXICAL_WEIGHT = 0.7
HYBRID_VECTOR_WEIGHT = 0.3


@dataclass(frozen=True)
class StoredPointPreview:
    id: int | str
    chunk_index: int
    text_preview: str


@dataclass(frozen=True)
class SearchResult:
    id: int | str
    score: float
    document_id: str
    chunk_index: int
    start: int
    end: int
    text: str
    text_preview: str
    page: int | None = None
    section_title: str | None = None


class InMemoryVectorStore:
    """Small Qdrant wrapper for vector storage and search."""

    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
        self.collection_name = collection_name

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance,
                FieldCondition,
                Filter,
                MatchValue,
                PointStruct,
                VectorParams,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Vector storage requires qdrant-client. Install it with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self._client = QdrantClient(":memory:")
        self._distance = Distance
        self._field_condition = FieldCondition
        self._filter = Filter
        self._match_value = MatchValue
        self._point_struct = PointStruct
        self._vector_params = VectorParams
        self._vector_size: int | None = None

    def store_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[EmbeddingResult],
        document_id: str,
    ) -> int:
        """Create a collection and store chunk vectors with payload metadata."""
        self._validate_inputs(chunks, embeddings)

        if not embeddings:
            return 0

        vector_size = embeddings[0].dimensions
        self._ensure_collection(vector_size)

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                self._point_struct(
                    id=_build_point_id(document_id, chunk.index),
                    vector=embedding.values,
                    payload={
                        "document_id": document_id,
                        "chunk_index": chunk.index,
                        "start": chunk.start,
                        "end": chunk.end,
                        "page": chunk.page,
                        "section_title": chunk.section_title,
                        "text": chunk.text,
                        "embedding_provider": embedding.provider,
                        "embedding_model": embedding.model,
                    },
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return len(points)

    def preview_points(
        self,
        limit: int = 3,
        document_id: str | None = None,
    ) -> list[StoredPointPreview]:
        """Read a few stored points back from Qdrant so we can verify storage."""
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            scroll_filter=self._document_filter(document_id),
            with_payload=True,
            with_vectors=False,
        )

        previews = []
        for record in records:
            payload = record.payload or {}
            text = str(payload.get("text", ""))
            previews.append(
                StoredPointPreview(
                    id=record.id,
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text_preview=_preview_text(text),
                )
            )

        return previews

    def count_document_points(self, document_id: str) -> int:
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            limit=10_000,
            scroll_filter=self._document_filter(document_id),
            with_payload=False,
            with_vectors=False,
        )

        return len(records)

    def delete_document(self, document_id: str) -> int:
        """Delete all stored vectors for one document and return deleted count."""
        deleted_count = self.count_document_points(document_id)

        if deleted_count == 0:
            return 0

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=self._document_filter(document_id),
            wait=True,
        )

        return deleted_count

    def search(
        self,
        query_vector: list[float],
        limit: int = 3,
        document_id: str | None = None,
        query_text: str | None = None,
        hybrid: bool = True,
    ) -> list[SearchResult]:
        """Find stored chunks with vector search, optionally reranked by local text search."""
        if limit <= 0:
            raise ValueError("search limit must be greater than 0.")

        candidate_limit = max(limit * 4, limit)
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._document_filter(document_id),
            limit=candidate_limit,
            with_payload=True,
            with_vectors=False,
        )

        if not hybrid or not query_text:
            return [
                _payload_to_search_result(point.id, point.payload or {}, point.score)
                for point in response.points[:limit]
            ]

        records = self._scroll_records(document_id)
        return _build_hybrid_search_results(
            query_text=query_text,
            vector_points=response.points,
            records=records,
            limit=limit,
        )

    def _scroll_records(self, document_id: str | None):
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            limit=10_000,
            scroll_filter=self._document_filter(document_id),
            with_payload=True,
            with_vectors=False,
        )

        return records

    def _ensure_collection(self, vector_size: int) -> None:
        if self._vector_size is not None and self._vector_size != vector_size:
            raise ValueError(
                "this vector store already contains "
                f"{self._vector_size}-dimension vectors; "
                f"got {vector_size}-dimension vectors."
            )

        if self._client.collection_exists(self.collection_name):
            self._vector_size = vector_size
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._vector_params(
                size=vector_size,
                distance=self._distance.COSINE,
            ),
        )
        self._vector_size = vector_size

    def _document_filter(self, document_id: str | None):
        if not document_id:
            return None

        return self._filter(
            must=[
                self._field_condition(
                    key="document_id",
                    match=self._match_value(value=document_id),
                )
            ]
        )

    def _validate_inputs(
        self,
        chunks: list[TextChunk],
        embeddings: list[EmbeddingResult],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")

        if not embeddings:
            return

        expected_dimensions = embeddings[0].dimensions
        for embedding in embeddings:
            if embedding.dimensions != expected_dimensions:
                raise ValueError("all embeddings must have the same dimensions.")


def _preview_text(text: str, limit: int = 90) -> str:
    compact_text = " ".join(text.split())

    if len(compact_text) <= limit:
        return compact_text

    return f"{compact_text[:limit].rstrip()}..."


def _build_point_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}"))


def _build_hybrid_search_results(
    query_text: str,
    vector_points,
    records,
    limit: int,
) -> list[SearchResult]:
    record_by_id = {_point_key(record.id): record for record in records}
    vector_scores = {
        _point_key(point.id): float(point.score)
        for point in vector_points
    }
    lexical_scores = _lexical_scores(query_text, records)

    normalized_vectors = _normalize_score_map(vector_scores)
    normalized_lexical = _normalize_score_map(lexical_scores)
    candidate_ids = set(normalized_vectors) | {
        point_id
        for point_id, score in lexical_scores.items()
        if score > 0
    }

    if not candidate_ids:
        candidate_ids = set(normalized_vectors)

    ranked_candidates = []
    for point_id in candidate_ids:
        record = record_by_id.get(point_id)

        if not record:
            continue

        payload = record.payload or {}
        score = (
            HYBRID_LEXICAL_WEIGHT * normalized_lexical.get(point_id, 0.0)
            + HYBRID_VECTOR_WEIGHT * normalized_vectors.get(point_id, 0.0)
        )
        ranked_candidates.append(
            (
                score,
                int(payload.get("chunk_index", 0)),
                record,
            )
        )

    ranked_candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    positive_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate[0] > 0
    ]

    if positive_candidates:
        ranked_candidates = positive_candidates

    return [
        _payload_to_search_result(record.id, record.payload or {}, score)
        for score, _, record in ranked_candidates[:limit]
    ]


def _lexical_scores(query_text: str, records) -> dict[str, float]:
    query_terms = list(dict.fromkeys(tokenize_for_local_search(query_text)))

    if not query_terms:
        return {}

    document_stats = []
    document_frequency = Counter()

    for record in records:
        payload = record.payload or {}
        text = str(payload.get("text", ""))
        section_title = str(payload.get("section_title", ""))
        tokens = tokenize_for_local_search(text)
        token_counts = Counter(tokens)
        document_stats.append((record, text, section_title, token_counts, len(tokens)))

        for term in query_terms:
            if token_counts.get(term, 0) > 0:
                document_frequency[term] += 1

    if not document_stats:
        return {}

    document_count = len(document_stats)
    average_document_length = (
        sum(length for _, _, _, _, length in document_stats) / document_count
    ) or 1.0
    scores: dict[str, float] = {}
    k1 = 1.5
    b = 0.75

    for record, text, section_title, token_counts, document_length in document_stats:
        score = 0.0
        normalized_length = max(document_length, 1)

        for term in query_terms:
            term_frequency = token_counts.get(term, 0)

            if term_frequency <= 0:
                continue

            term_document_frequency = document_frequency[term]
            inverse_document_frequency = math.log(
                1
                + (
                    document_count
                    - term_document_frequency
                    + 0.5
                )
                / (term_document_frequency + 0.5)
            )
            denominator = term_frequency + k1 * (
                1 - b + b * normalized_length / average_document_length
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * (k1 + 1)
                / denominator
            )

        matched_terms = sum(
            1
            for term in query_terms
            if token_counts.get(term, 0) > 0
        )

        if matched_terms:
            score += 0.35 * matched_terms / len(query_terms)

        if _has_section_label_match(query_terms, text):
            score += 0.65

        score += _metadata_aware_bonus(query_terms, text, section_title)

        if score > 0:
            scores[_point_key(record.id)] = score

    return scores


def _has_section_label_match(query_terms: list[str], text: str) -> bool:
    lowered_text = text.lower()
    section_terms = {
        "framework": ("framework", "frameworks"),
        "library": ("library", "libraries"),
        "skill": ("skill", "skills"),
        "tool": ("tool", "tools"),
        "technology": ("technology", "technologies"),
    }

    for query_term, text_terms in section_terms.items():
        if query_term in query_terms and any(term in lowered_text for term in text_terms):
            return True

    return False


def _metadata_aware_bonus(
    query_terms: list[str],
    text: str,
    section_title: str,
) -> float:
    lowered_text = text.lower()
    lowered_section = section_title.lower()
    bonus = 0.0

    if "professional experience" in lowered_section and any(
        term in query_terms
        for term in ("work", "company", "role", "job")
    ):
        bonus += 0.45

    if "present" in lowered_text and any(
        term in query_terms
        for term in ("current", "currently", "work", "company", "role")
    ):
        bonus += 0.75

    if "technical skills" in lowered_section and any(
        term in query_terms
        for term in ("database", "framework", "library", "tool", "skill")
    ):
        bonus += 0.35

    return bonus


def _normalize_score_map(scores: dict[str, float]) -> dict[str, float]:
    positive_scores = {
        point_id: max(score, 0.0)
        for point_id, score in scores.items()
    }
    highest_score = max(positive_scores.values(), default=0.0)

    if highest_score <= 0:
        return {
            point_id: 0.0
            for point_id in positive_scores
        }

    return {
        point_id: score / highest_score
        for point_id, score in positive_scores.items()
    }


def _payload_to_search_result(point_id, payload: dict, score: float) -> SearchResult:
    text = str(payload.get("text", ""))
    page = payload.get("page")

    return SearchResult(
        id=point_id,
        score=float(score),
        document_id=str(payload.get("document_id", "")),
        chunk_index=int(payload.get("chunk_index", 0)),
        start=int(payload.get("start", 0)),
        end=int(payload.get("end", 0)),
        text=text,
        text_preview=_preview_text(text),
        page=int(page) if page is not None else None,
        section_title=payload.get("section_title"),
    )


def _point_key(point_id) -> str:
    return str(point_id)
