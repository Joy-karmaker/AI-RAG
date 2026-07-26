from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

from rag.vector_store import SearchResult
from rag.verification import VerificationResult


# Rough characters-per-token estimate used only for debug prompt sizing.
# The real Gemini tokenizer is not called here; this keeps the trace offline.
CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class TraceStage:
    name: str
    duration_ms: float


@dataclass
class QueryTrace:
    """Collects a debuggable record of one query as it moves through the pipeline.

    The trace stays deterministic and offline: it only measures latency and
    reads values the pipeline already produced (scores, prompt size, model,
    verification). It never calls a tokenizer or an external service.
    """

    query: str
    document_id: str
    filename: str = ""
    fields: dict = field(default_factory=dict)
    stages: list[TraceStage] = field(default_factory=list)

    def set(self, key: str, value: object) -> None:
        self.fields[key] = value

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a pipeline stage and store its duration in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.stages.append(TraceStage(name=name, duration_ms=duration_ms))

    def record_embedding(self, provider: str, model: str, dimensions: int) -> None:
        self.set(
            "embedding",
            {
                "provider": provider,
                "model": model,
                "dimensions": dimensions,
            },
        )

    def record_retrieval(
        self,
        query_mode: str,
        reranker: str,
        top_k: int,
        candidate_count: int,
        results: list[SearchResult],
    ) -> None:
        query_type = results[0].query_type if results else None
        self.set(
            "retrieval",
            {
                "query_mode": query_mode,
                "query_type": query_type,
                "reranker": reranker,
                "top_k": top_k,
                "candidate_count": candidate_count,
                "selected_count": len(results),
                "selected_chunks": [
                    {
                        "chunk_index": result.chunk_index,
                        "score": _round(result.score),
                        "vector_score": _round(result.vector_score),
                        "lexical_score": _round(result.lexical_score),
                        "rerank_score": _round(result.rerank_score),
                        "rerank_strategy": result.rerank_strategy,
                        "page": result.page,
                        "section_title": result.section_title,
                    }
                    for result in results
                ],
            },
        )

    def record_prompt(self, prompt: str, model: Optional[str]) -> None:
        self.set(
            "prompt",
            {
                "chars": len(prompt),
                "tokens_estimate": _estimate_tokens(prompt),
                "model": model,
            },
        )

    def record_answer(self, answer: str, model: str) -> None:
        self.set(
            "answer",
            {
                "model": model,
                "chars": len(answer),
            },
        )

    def record_verification(self, verification: VerificationResult) -> None:
        self.set(
            "verification",
            {
                "status": verification.verification_status,
                "supported": verification.supported,
                "answer_coverage": verification.answer_coverage,
                "evidence_score": verification.evidence_score,
                "cited_chunks": verification.cited_chunks,
            },
        )

    def total_latency_ms(self) -> float:
        return round(sum(stage.duration_ms for stage in self.stages), 2)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "document_id": self.document_id,
            "filename": self.filename,
            **self.fields,
            "stages": [
                {"name": stage.name, "duration_ms": stage.duration_ms}
                for stage in self.stages
            ],
            "latency_ms": {
                "total": self.total_latency_ms(),
            },
        }

    def to_text(self) -> str:
        """Render the compact debug view shown in the CLI and API docs."""
        lines = ["=== RAG Trace ===", f"Query: {self.query}"]

        if self.filename:
            lines.append(f"Document: {self.filename} ({self.document_id})")
        else:
            lines.append(f"Document: {self.document_id}")

        embedding = self.fields.get("embedding")
        if embedding:
            lines.append(
                "Embedding: "
                f"{embedding['provider']} / {embedding['model']} "
                f"({embedding['dimensions']} dims)"
            )

        retrieval = self.fields.get("retrieval")
        if retrieval:
            lines.append(
                "Retriever candidates: "
                f"{retrieval['candidate_count']} "
                f"(query_type: {retrieval['query_type']}, "
                f"mode: {retrieval['query_mode']})"
            )
            selected = retrieval["selected_chunks"]
            kept = ", ".join(str(chunk["chunk_index"]) for chunk in selected) or "-"
            lines.append(f"Reranker ({retrieval['reranker']}) selected: chunks {kept}")
            for chunk in selected:
                lines.append(
                    f"  - chunk {chunk['chunk_index']}: "
                    f"score {_format(chunk['score'])}, "
                    f"vector {_format(chunk['vector_score'])}, "
                    f"lexical {_format(chunk['lexical_score'])}, "
                    f"rerank {_format(chunk['rerank_score'])}"
                )

        prompt = self.fields.get("prompt")
        if prompt:
            lines.append(
                f"Prompt: {prompt['chars']} chars "
                f"(~{prompt['tokens_estimate']} tokens)"
            )

        answer = self.fields.get("answer")
        if answer:
            lines.append(f"Answer model: {answer['model']}")

        verification = self.fields.get("verification")
        if verification:
            lines.append(f"Verification: {verification['status']}")

        for stage in self.stages:
            lines.append(f"Stage {stage.name}: {stage.duration_ms} ms")

        lines.append(f"Latency: {self.total_latency_ms()} ms")
        return "\n".join(lines)


def timed(trace: Optional[QueryTrace], name: str, action):
    """Run ``action`` and time it as a named stage when a trace is active.

    Keeps callers free of ``if trace is not None`` branching around every stage.
    """
    if trace is None:
        return action()

    with trace.stage(name):
        return action()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0

    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    return round(float(value), 4)


def _format(value: Optional[float]) -> str:
    if value is None:
        return "-"

    return f"{value:.4f}"
