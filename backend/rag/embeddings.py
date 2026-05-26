from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
DEFAULT_LOCAL_DIMENSIONS = 64
DEFAULT_GEMINI_MODEL = "gemini-embedding-2"


@dataclass(frozen=True)
class EmbeddingResult:
    index: int
    values: list[float]
    provider: str
    model: str
    source_characters: int

    @property
    def dimensions(self) -> int:
        return len(self.values)

    @property
    def magnitude(self) -> float:
        return vector_magnitude(self.values)


def embed_texts(
    texts: Iterable[str],
    provider: str = "local",
    dimensions: int = DEFAULT_LOCAL_DIMENSIONS,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> list[EmbeddingResult]:
    """Turn text strings into numeric vectors."""
    text_list = list(texts)

    if provider == "local":
        return _embed_with_local_hashing(text_list, dimensions=dimensions)

    if provider == "gemini":
        return _embed_with_gemini(text_list, model=gemini_model)

    raise ValueError("embedding provider must be 'local' or 'gemini'.")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cannot compare vectors with different dimensions.")

    left_magnitude = vector_magnitude(left)
    right_magnitude = vector_magnitude(right)

    if left_magnitude == 0 or right_magnitude == 0:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_magnitude * right_magnitude)


def vector_magnitude(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def format_vector_preview(values: list[float], limit: int = 8) -> str:
    preview = ", ".join(f"{value:.4f}" for value in values[:limit])

    if len(values) > limit:
        return f"[{preview}, ...]"

    return f"[{preview}]"


def _embed_with_local_hashing(texts: list[str], dimensions: int) -> list[EmbeddingResult]:
    if dimensions <= 0:
        raise ValueError("embedding dimensions must be greater than 0.")

    return [
        EmbeddingResult(
            index=index,
            values=_local_hash_embedding(text, dimensions),
            provider="local",
            model=f"local-hash-{dimensions}",
            source_characters=len(text),
        )
        for index, text in enumerate(texts, start=1)
    ]


def _local_hash_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = TOKEN_PATTERN.findall(text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], byteorder="big") % dimensions
        weight = 1.0 + min(len(token), 12) / 12
        vector[bucket] += weight

    return _normalize(vector)


def _normalize(values: list[float]) -> list[float]:
    magnitude = vector_magnitude(values)

    if magnitude == 0:
        return values

    return [value / magnitude for value in values]


def _embed_with_gemini(texts: list[str], model: str) -> list[EmbeddingResult]:
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Gemini embeddings require an API key. Set GEMINI_API_KEY first."
        )

    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gemini embeddings require google-genai. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    client = genai.Client()
    results: list[EmbeddingResult] = []

    for index, text in enumerate(texts, start=1):
        if model == "gemini-embedding-001":
            response = client.models.embed_content(
                model=model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
        else:
            response = client.models.embed_content(
                model=model,
                contents=f"title: none | text: {text}",
            )

        values = list(response.embeddings[0].values)
        results.append(
            EmbeddingResult(
                index=index,
                values=values,
                provider="gemini",
                model=model,
                source_characters=len(text),
            )
        )

    return results
