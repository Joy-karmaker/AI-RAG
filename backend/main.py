from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.chunking import chunk_text
from rag.embeddings import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LOCAL_DIMENSIONS,
    cosine_similarity,
    embed_texts,
    format_vector_preview,
)
from rag.extractor import extract_file_text
from rag.generation import DEFAULT_LLM_MODEL, generate_grounded_answer
from rag.prompt import build_grounded_prompt
from rag.vector_store import DEFAULT_COLLECTION_NAME, InMemoryVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Day 6: retrieve relevant chunks and ask Gemini for a grounded answer."
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
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Create embeddings for each text chunk.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("local", "gemini"),
        default="local",
        help="Embedding provider to use with --embed. Default: local.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=DEFAULT_LOCAL_DIMENSIONS,
        help=f"Dimensions for local embeddings. Default: {DEFAULT_LOCAL_DIMENSIONS}.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini embedding model to use. Default: {DEFAULT_GEMINI_MODEL}.",
    )
    parser.add_argument(
        "--store-vectors",
        action="store_true",
        help="Store chunk embeddings in an in-memory Qdrant collection.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name. Default: {DEFAULT_COLLECTION_NAME}.",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Document id saved in each vector payload. Default: file name.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Question/search text to retrieve relevant chunks from Qdrant.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of search results to return. Default: 3.",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Use Gemini to answer the query from retrieved chunks.",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"Gemini model for answer generation. Default: {DEFAULT_LLM_MODEL}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Gemini answer temperature. Lower is more focused. Default: 0.2.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the grounded prompt sent to Gemini.",
    )
    parser.add_argument(
        "--dry-run-answer",
        action="store_true",
        help="Build and print the grounded prompt without calling Gemini.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        text = extract_file_text(args.file_path)
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        embeddings = []
        stored_count = 0
        stored_previews = []
        search_results = []
        query_embedding = None
        grounded_answer = None
        grounded_prompt = None

        if (args.answer or args.dry_run_answer) and not args.query:
            raise ValueError("--answer and --dry-run-answer require --query.")

        should_create_embeddings = (
            args.embed
            or args.store_vectors
            or args.query
            or args.answer
            or args.dry_run_answer
        )
        if should_create_embeddings:
            embeddings = embed_texts(
                [chunk.text for chunk in chunks],
                provider=args.embedding_provider,
                dimensions=args.embedding_dimensions,
                gemini_model=args.gemini_model,
            )

        should_use_vector_store = (
            args.store_vectors or args.query or args.answer or args.dry_run_answer
        )
        if should_use_vector_store:
            document_id = args.document_id or args.file_path.name
            vector_store = InMemoryVectorStore(collection_name=args.collection_name)
            stored_count = vector_store.store_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_id=document_id,
            )
            stored_previews = vector_store.preview_points()

            if args.query:
                query_embedding = embed_texts(
                    [args.query],
                    provider=args.embedding_provider,
                    dimensions=args.embedding_dimensions,
                    gemini_model=args.gemini_model,
                )[0]
                search_results = vector_store.search(
                    query_vector=query_embedding.values,
                    limit=args.top_k,
                    document_id=document_id,
                    query_text=args.query,
                )

                if args.dry_run_answer:
                    grounded_prompt = build_grounded_prompt(args.query, search_results)
                elif args.answer:
                    grounded_answer = generate_grounded_answer(
                        question=args.query,
                        search_results=search_results,
                        model=args.llm_model,
                        temperature=args.temperature,
                    )
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

    if args.embed or args.store_vectors or args.query or args.answer or args.dry_run_answer:
        print("\n=== Embeddings ===")

        for embedding in embeddings:
            print(f"\n--- Chunk {embedding.index} Vector ---")
            print(f"Provider: {embedding.provider}")
            print(f"Model: {embedding.model}")
            print(f"Dimensions: {embedding.dimensions}")
            print(f"Magnitude: {embedding.magnitude:.4f}")
            print(f"Preview: {format_vector_preview(embedding.values)}")

        if len(embeddings) >= 2:
            similarity = cosine_similarity(embeddings[0].values, embeddings[1].values)
            print("\n=== Similarity Check ===")
            print(f"Chunk 1 vs Chunk 2 cosine similarity: {similarity:.4f}")

    if args.store_vectors or args.query or args.answer or args.dry_run_answer:
        print("\n=== Vector Store ===")
        print("Storage: Qdrant in-memory")
        print(f"Collection: {args.collection_name}")
        print(f"Points stored: {stored_count}")

        if stored_previews:
            print("\nStored point preview:")
            for preview in stored_previews:
                print(
                    f"- Point {preview.id}: chunk {preview.chunk_index}, "
                    f"{preview.text_preview}"
                )

    if args.query:
        print("\n=== Hybrid Search ===")
        print(f"Query: {args.query}")

        if query_embedding:
            print(f"Query vector: {format_vector_preview(query_embedding.values)}")

        if search_results:
            print(f"\nTop {len(search_results)} result(s):")
            for rank, result in enumerate(search_results, start=1):
                print(
                    f"\n{rank}. Chunk {result.chunk_index} "
                    f"(score: {result.score:.4f}, chars: {result.start}-{result.end})"
                )
                print(result.text_preview)
        else:
            print("No matching chunks found.")

    if args.dry_run_answer and grounded_prompt:
        print("\n=== Grounded Prompt ===")
        print(grounded_prompt)
        print("\nGemini call skipped because --dry-run-answer was used.")

    if args.answer and grounded_answer:
        if args.show_prompt:
            print("\n=== Grounded Prompt ===")
            print(grounded_answer.prompt)

        print("\n=== Gemini Answer ===")
        print(f"Model: {grounded_answer.model}")
        print(f"Context chunks used: {grounded_answer.context_count}")
        print()
        print(grounded_answer.answer)

    print("\n=== Summary ===")
    print(f"Characters extracted: {len(text)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap: {args.overlap}")

    if args.embed or args.store_vectors or args.query or args.answer or args.dry_run_answer:
        print(f"Embeddings created: {len(embeddings)}")
    if args.store_vectors or args.query or args.answer or args.dry_run_answer:
        print(f"Vectors stored: {stored_count}")
    if args.query:
        print(f"Search results: {len(search_results)}")
    if args.answer and grounded_answer:
        print("Gemini answer generated: yes")


if __name__ == "__main__":
    main()
