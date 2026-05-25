from __future__ import annotations

from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md"}


def extract_file_text(file_path: str | Path) -> str:
    """Extract raw text from a supported local document."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file path, got: {path}")

    extension = path.suffix.lower()

    if extension in TEXT_EXTENSIONS:
        return _extract_text_file(path)

    if extension == ".pdf":
        return _extract_pdf_file(path)

    supported = ", ".join(sorted([*TEXT_EXTENSIONS, ".pdf"]))
    raise ValueError(f"Unsupported file type '{extension}'. Supported: {supported}")


def _extract_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path} as utf-8, utf-8-sig, or cp1252.",
    )


def _extract_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF extraction requires pypdf. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(f"--- Page {page_number} ---\n{page_text.strip()}")

    return "\n\n".join(pages).strip()
