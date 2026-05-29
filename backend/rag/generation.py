from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rag.config import get_gemini_api_key, load_env_file
from rag.prompt import GROUNDING_SYSTEM_INSTRUCTION, build_grounded_prompt
from rag.vector_store import SearchResult


DEFAULT_LLM_MODEL = "gemini-2.5-flash"


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

    answer = _generate_with_gemini_rest(
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


def _generate_with_gemini_rest(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
) -> str:
    safe_model = urllib.parse.quote(model, safe="")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{safe_model}:generateContent"
    )
    payload = {
        "systemInstruction": {
            "parts": [{"text": GROUNDING_SYSTEM_INSTRUCTION}],
        },
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini API request failed with HTTP {exc.code}: "
            f"{_extract_gemini_error_message(body)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Gemini API returned an unexpected response: {data}") from exc


def _extract_gemini_error_message(body: str) -> str:
    try:
        data = json.loads(body)
        return str(data["error"]["message"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return body[:500]
