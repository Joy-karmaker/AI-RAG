from __future__ import annotations

import os
from pathlib import Path


GEMINI_API_KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY")


def load_env_file(env_path: str | Path = ".env") -> None:
    """Load simple KEY=value lines from a .env file without printing secrets."""
    path = Path(env_path)

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value)

        if key and key not in os.environ:
            os.environ[key] = value


def get_gemini_api_key() -> str | None:
    for key_name in GEMINI_API_KEY_NAMES:
        api_key = os.environ.get(key_name)

        if api_key:
            return api_key

    return None


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]

    return cleaned
