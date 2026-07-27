from __future__ import annotations

import json
import os
from pathlib import Path


class DocumentRegistry:
    """Persist document metadata separately from the Qdrant vector payloads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, dict[str, object]]:
        if not self.path.exists():
            return {}

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read persisted document metadata from {self.path}."
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Persisted document metadata must be a JSON object.")

        return {
            str(document_id): record
            for document_id, record in payload.items()
            if isinstance(record, dict)
        }

    def save(self, records: dict[str, dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.path)
