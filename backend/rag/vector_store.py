from __future__ import annotations

from dataclasses import dataclass

from rag.chunking import TextChunk
from rag.embeddings import EmbeddingResult


DEFAULT_COLLECTION_NAME = "ai_rag_chunks"


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


class InMemoryVectorStore:
    """Small Qdrant wrapper for vector storage and search."""

    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
        self.collection_name = collection_name

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Vector storage requires qdrant-client. Install it with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self._client = QdrantClient(":memory:")
        self._distance = Distance
        self._point_struct = PointStruct
        self._vector_params = VectorParams

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
        self._create_collection(vector_size)

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                self._point_struct(
                    id=chunk.index,
                    vector=embedding.values,
                    payload={
                        "document_id": document_id,
                        "chunk_index": chunk.index,
                        "start": chunk.start,
                        "end": chunk.end,
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

    def preview_points(self, limit: int = 3) -> list[StoredPointPreview]:
        """Read a few stored points back from Qdrant so we can verify storage."""
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            limit=limit,
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

    def search(self, query_vector: list[float], limit: int = 3) -> list[SearchResult]:
        """Find stored chunks whose vectors are closest to the query vector."""
        if limit <= 0:
            raise ValueError("search limit must be greater than 0.")

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            results.append(
                SearchResult(
                    id=point.id,
                    score=point.score,
                    document_id=str(payload.get("document_id", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    start=int(payload.get("start", 0)),
                    end=int(payload.get("end", 0)),
                    text=text,
                    text_preview=_preview_text(text),
                )
            )

        return results

    def _create_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._vector_params(
                size=vector_size,
                distance=self._distance.COSINE,
            ),
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
