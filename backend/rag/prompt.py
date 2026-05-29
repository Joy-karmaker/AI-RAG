from __future__ import annotations

from rag.vector_store import SearchResult


GROUNDING_SYSTEM_INSTRUCTION = (
    "You are a careful RAG assistant. Answer using only the provided context. "
    "If the context does not contain the answer, say that the document context "
    "does not provide enough information."
)


def build_grounded_prompt(question: str, search_results: list[SearchResult]) -> str:
    if not question.strip():
        raise ValueError("question cannot be empty.")

    if not search_results:
        raise ValueError("at least one search result is required to build a prompt.")

    context = _format_context(search_results)

    return f"""Use the document context below to answer the user's question.

Rules:
- Answer only from the context.
- Do not invent facts.
- If the answer is not in the context, say: "The provided document context does not contain enough information to answer that."
- Keep the answer concise and clear.
- End with a short Sources line listing the chunk numbers used.

Document context:
{context}

Question:
{question.strip()}

Answer:"""


def _format_context(search_results: list[SearchResult]) -> str:
    blocks = []

    for rank, result in enumerate(search_results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Source {rank}]",
                    f"Document: {result.document_id}",
                    f"Chunk: {result.chunk_index}",
                    f"Similarity score: {result.score:.4f}",
                    "Text:",
                    result.text.strip(),
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)
