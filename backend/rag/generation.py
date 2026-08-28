from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rag.config import get_gemini_api_key, load_env_file
from rag.prompt import GROUNDING_SYSTEM_INSTRUCTION, build_grounded_prompt
from rag.vector_store import SearchResult


DEFAULT_LLM_MODEL = "gemini-2.5-flash"
MAX_GENERATION_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.75


@dataclass(frozen=True)
class GroundedAnswer:
    question: str
    answer: str
    model: str
    prompt: str
    context_count: int


def generate_grounded_answer(
    question: str,
    search_results: list[SearchResult],
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
    env_path: str | Path = ".env",
) -> GroundedAnswer:
    """Build a grounded prompt and ask Gemini to answer from retrieved context."""
    load_env_file(env_path)
    api_key = get_gemini_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini answer generation requires an API key in GEMINI_API_KEY, "
            "GOOGLE_API_KEY, or API_KEY."
        )

    prompt = build_grounded_prompt(question, search_results)

    answer = _generate_with_gemini(
        api_key=api_key,
        model=model,
        prompt=prompt,
        temperature=temperature,
    )

    if not answer:
        answer = "Gemini returned an empty answer."

    return GroundedAnswer(
        question=question,
        answer=answer,
        model=model,
        prompt=prompt,
        context_count=len(search_results),
    )


def _generate_with_gemini(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
) -> str:
    """Call Gemini through the google-genai SDK with simple retry/backoff."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Gemini answer generation requires a working google-genai installation. "
            "Install or repair dependencies with: python -m pip install -r requirements.txt. "
            f"Import error: {exc}"
        ) from exc

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=GROUNDING_SYSTEM_INSTRUCTION,
        temperature=temperature,
    )

    last_error: Exception | None = None
    for attempt in range(1, MAX_GENERATION_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            answer = (response.text or "").strip()
            return answer
        except Exception as exc:  # noqa: BLE001 - SDK surfaces varied error types
            last_error = exc
            if attempt < MAX_GENERATION_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"Gemini generation failed after {MAX_GENERATION_RETRIES} attempts: {last_error}")
