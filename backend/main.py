from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.chunking import chunk_text
from rag.extractor import extract_file_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Day 2: extract text from a document and split it into chunks."
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to a .txt, .md, or .pdf document.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per chunk. Default: 1000.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Characters repeated between neighboring chunks. Default: 200.",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print the full extracted document text before chunking.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        text = extract_file_text(args.file_path)
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
    except (FileNotFoundError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.show_text:
        print("\n=== Document Text ===\n")
        print(text)

    print("\n=== Text Chunks ===")

    for chunk in chunks:
        print(f"\n--- Chunk {chunk.index} ({chunk.start}-{chunk.end}) ---")
        print(chunk.text)

    print("\n=== Summary ===")
    print(f"Characters extracted: {len(text)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap: {args.overlap}")


if __name__ == "__main__":
    main()
