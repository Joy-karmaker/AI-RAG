from __future__ import annotations

import json
import sys
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path


API_BASE = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOC = PROJECT_ROOT / "sample_docs" / "day2_long.txt"


def main() -> None:
    print("AI-RAG smoke test")
    print(f"API: {API_BASE}")

    root = get_json("/")
    assert_equal(root["version"], "1.0.0", "API version")

    status = get_json("/system/status")
    assert_equal(status["status"], "ok", "system status")
    assert_equal(status["frontend_available"], True, "frontend availability")

    app_html = get_text("/app/")
    assert_contains(app_html, "AI-RAG", "frontend HTML")

    document_id = f"smoke-{uuid.uuid4().hex[:8]}"
    upload = upload_document(document_id)
    assert_equal(upload["document_id"], document_id, "uploaded document id")
    assert_greater(upload["vector_count"], 0, "stored vector count")

    query = post_json(
        f"/documents/{document_id}/query",
        {
            "query": "Why do chunks overlap?",
            "top_k": 2,
            "answer": False,
        },
    )
    assert_equal(query["document_id"], document_id, "query document id")
    assert_greater(len(query["results"]), 0, "search result count")

    prompt = post_json(
        f"/documents/{document_id}/query",
        {
            "query": "Why do chunks overlap?",
            "top_k": 2,
            "dry_run_answer": True,
        },
    )
    assert_contains(prompt["prompt"], "Document context", "grounded prompt")

    deleted = delete_json(f"/documents/{document_id}")
    assert_equal(deleted["document_id"], document_id, "deleted document id")
    assert_greater(deleted["vectors_deleted"], 0, "deleted vector count")

    documents = get_json("/documents")
    remaining_ids = {document["document_id"] for document in documents["documents"]}
    if document_id in remaining_ids:
        raise SystemExit("document delete failed: deleted document still listed")
    print("ok: deleted document removed from list")

    print("Smoke test passed.")


def get_json(path: str) -> dict:
    return request_json("GET", path)


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    return request_json("POST", path, data=data, headers=headers)


def delete_json(path: str) -> dict:
    return request_json("DELETE", path)


def get_text(path: str) -> str:
    request = urllib.request.Request(f"{API_BASE}{path}", method="GET")

    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read().decode("utf-8")


def request_json(
    method: str,
    path: str,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict:
    request = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers=headers or {},
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed for {path}: {exc}") from exc


def upload_document(document_id: str) -> dict:
    boundary = f"----ai-rag-{uuid.uuid4().hex}"
    body = build_multipart_body(
        boundary=boundary,
        fields={
            "document_id": document_id,
            "chunk_size": "600",
            "overlap": "120",
        },
        file_field="file",
        file_path=SAMPLE_DOC,
    )
    request = urllib.request.Request(
        f"{API_BASE}/documents/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def build_multipart_body(
    boundary: str,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> bytes:
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "utf-8"
                ),
                f"{value}\r\n".encode("utf-8"),
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: text/plain\r\n\r\n",
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    return b"".join(chunks)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise SystemExit(f"{label} failed: expected {expected!r}, got {actual!r}")

    print(f"ok: {label}")


def assert_greater(actual: int, minimum: int, label: str) -> None:
    if actual <= minimum:
        raise SystemExit(f"{label} failed: expected > {minimum}, got {actual}")

    print(f"ok: {label}")


def assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"{label} failed: missing {needle!r}")

    print(f"ok: {label}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
